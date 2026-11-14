"""Five-class plant classification: train, validate, refit, and predict.

Example: python plant_classifier.py run --zip /path/2026task1.zip --out results
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def unpack(zip_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Archive path escapes destination')
        archive.extractall(destination)
    return destination


def locate_data(path):
    path = Path(path)
    candidates = [path] + sorted(p.parent for p in path.rglob('train') if p.is_dir())
    for p in candidates:
        if (p / 'train').is_dir() and (p / 'test').is_dir():
            return p
    raise FileNotFoundError('Expected train/<class>/*.png and test/*.png')


def locate_template(path):
    choices = sorted(Path(path).rglob('submission-for-task2.csv'))
    if not choices:
        raise FileNotFoundError('Pass --template with the original sample submission')
    return choices[0]


def collect(root):
    classes = sorted(p.name for p in (root / 'train').iterdir() if p.is_dir())
    records = [(p, i) for i, c in enumerate(classes)
               for p in sorted((root / 'train' / c).glob('*.png'))]
    if len(classes) != 5 or not records:
        raise ValueError(f'Expected five plant classes, got {classes}')
    return classes, records


def stratified_split(records, seed, fraction=0.2):
    rng = np.random.default_rng(seed)
    train, valid = [], []
    for label in sorted({r[1] for r in records}):
        members = [r for r in records if r[1] == label]
        order = rng.permutation(len(members))
        n = max(1, round(len(members) * fraction))
        valid.extend(members[i] for i in order[:n])
        train.extend(members[i] for i in order[n:])
    # Exact duplicates would make an image-level validation optimistic.
    hashes = lambda rows: {hashlib.sha256(p.read_bytes()).hexdigest() for p, _ in rows}
    if hashes(train) & hashes(valid):
        raise ValueError('Exact duplicates cross the training/validation split')
    return train, valid


class QuarterTurn:
    def __call__(self, image):
        turns = random.randrange(4)
        return image if turns == 0 else image.transpose([
            Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270][turns - 1])


class PlantDataset(Dataset):
    def __init__(self, records, size, training=False):
        self.records = records
        # Cache a modest resolution rather than decoding large PNGs each epoch.
        self.images = []
        for p, _ in records:
            with Image.open(p) as im:
                self.images.append(im.convert('RGB').resize(
                    (size + 32, size + 32), Image.Resampling.BILINEAR))
        operations = [T.RandomResizedCrop(size, scale=(0.80, 1.0), ratio=(0.9, 1.1)),
                      QuarterTurn(), T.RandomHorizontalFlip(),
                      T.RandomRotation(35, interpolation=T.InterpolationMode.BILINEAR),
                      T.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.025)
                      ] if training else [T.Resize((size, size))]
        self.transform = T.Compose(operations + [T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.records[index][1]


def build_model(classes, pretrained=True, architecture='efficientnet_b0'):
    constructor, weights_type = {
        'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights),
        'efficientnet_b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights)
    }[architecture]
    weights = weights_type.IMAGENET1K_V1 if pretrained else None
    model = constructor(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    return model


@torch.inference_mode()
def probabilities(model, loader, device, tta=False):
    model.eval()
    outputs = []
    for x, _ in loader:
        x = x.to(device)
        variants = [x]
        if tta:
            variants = [torch.rot90(x, k, (-2, -1)) for k in range(4)]
            variants += [v.flip(-1) for v in variants.copy()]
        accum = torch.zeros((len(x), model.classifier[1].out_features), device=device)
        for v in variants:
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(v)
            accum += logits.float().softmax(-1)
        outputs.append((accum / len(variants)).cpu().numpy())
    return np.concatenate(outputs)


def classification_metrics(labels, probs, classes):
    pred = probs.argmax(axis=1)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(cm, (labels, pred), 1)
    tp = cm.diagonal().astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=precision + recall > 0)
    acc = float((pred == labels).mean())
    return {'accuracy': acc, 'micro_f1': acc, 'macro_f1': float(f1.mean()),
            'n': len(labels), 'confusion_matrix': cm.tolist(),
            'per_class': {c: {'precision': float(precision[i]), 'recall': float(recall[i]),
                              'f1': float(f1[i]), 'support': int(cm[i].sum())}
                          for i, c in enumerate(classes)}}


def checkpoint(model, classes, args, epoch):
    return {'architecture': args.architecture, 'classes': classes, 'size': args.size,
            'epoch': epoch, 'seed': args.seed, 'mean': MEAN, 'std': STD,
            'tta': args.tta,
            'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}}


def fit(records, valid_records, classes, args, destination, device):
    destination.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    model = build_model(classes, architecture=args.architecture).to(device)
    train_set = PlantDataset(records, args.size, training=True)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=device.type == 'cuda')
    validation = None
    if valid_records:
        validation = DataLoader(PlantDataset(valid_records, args.size),
                                batch_size=args.batch_size, num_workers=0)
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr * 3}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best_score, best_loss, best_epoch = -1., float('inf'), 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, seen = 0., 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(y)
            correct += int((logits.argmax(-1) == y).sum())
            seen += len(y)
        scheduler.step()
        row = {'epoch': epoch, 'train_loss': loss_sum / seen,
               'train_accuracy_augmented': correct / seen,
               'elapsed_seconds': round(time.time() - started, 2)}
        if validation:
            probs = probabilities(model, validation, device, tta=args.tta)
            labels = np.array([r[1] for r in valid_records])
            score = float((probs.argmax(1) == labels).mean())
            val_loss = float(-np.log(probs[np.arange(len(labels)), labels].clip(1e-8)).mean())
            row.update(val_micro_f1=score, val_loss=val_loss)
            if score > best_score or (score == best_score and val_loss < best_loss):
                best_score, best_loss, best_epoch = score, val_loss, epoch
                torch.save(checkpoint(model, classes, args, epoch), destination / 'model.pt')
        else:
            best_epoch = epoch
        history.append(row)
        save_json(destination / 'history.json', history)
        print(json.dumps(row), flush=True)
    if not validation:
        torch.save(checkpoint(model, classes, args, args.epochs), destination / 'model.pt')
    else:
        model.load_state_dict(torch.load(destination / 'model.pt', map_location=device,
                                         weights_only=True)['state_dict'])
        probs = probabilities(model, validation, device, tta=args.tta)
        labels = np.array([r[1] for r in valid_records])
        result = classification_metrics(labels, probs, classes)
        result.update(best_epoch=best_epoch, best_selection_micro_f1=best_score,
                      seed=args.seed, train_count=len(records), tta_views=8 if args.tta else 1)
        save_json(destination / 'metrics.json', result)
        with (destination / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'true_class', 'predicted_class', 'confidence'] + classes)
            for (p, label), prob in zip(valid_records, probs):
                writer.writerow([p.name, classes[label], classes[int(prob.argmax())],
                                 float(prob.max())] + prob.tolist())
        print('VALIDATION ' + json.dumps(result), flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_epoch


def predict(root, template, checkpoints, out, device, batch_size):
    with Path(template).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['ID', 'Category']:
            raise ValueError(f'Unexpected submission columns: {reader.fieldnames}')
        rows = list(reader)
    ids = [r['ID'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate test IDs')
    if set(ids) != {p.name for p in (root / 'test').glob('*.png')}:
        raise ValueError('Template IDs do not match test filenames')
    records = [(root / 'test' / name, 0) for name in ids]
    all_probs, classes, input_size, loader = [], None, None, None
    for path in checkpoints:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        if classes is not None and (classes != ckpt['classes'] or input_size != ckpt['size']):
            raise ValueError('Incompatible ensemble checkpoints')
        classes, input_size = ckpt['classes'], ckpt['size']
        if loader is None:
            loader = DataLoader(PlantDataset(records, input_size), batch_size=batch_size)
        model = build_model(classes, pretrained=False, architecture=ckpt['architecture']).to(device)
        model.load_state_dict(ckpt['state_dict'])
        all_probs.append(probabilities(model, loader, device, tta=ckpt.get('tta', True)))
        del model
    probs = np.mean(all_probs, axis=0)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'submission.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Category'])
        writer.writerows((name, classes[int(p.argmax())]) for name, p in zip(ids, probs))
    with (out / 'test_probabilities.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID'] + classes)
        writer.writerows([name] + p.tolist() for name, p in zip(ids, probs))
    save_json(out / 'submission_check.json', {
        'rows': len(ids), 'columns': ['ID', 'Category'], 'unique_ids': True,
        'matches_template_order': True, 'all_test_files_covered': True,
        'class_counts': {c: int((probs.argmax(1) == i).sum()) for i, c in enumerate(classes)},
        'checkpoints': [str(p) for p in checkpoints], 'tta_views_per_model': 8 if ckpt.get('tta', True) else 1,
        'sha256': hashlib.sha256((out / 'submission.csv').read_bytes()).hexdigest()})
    print(f'Submission saved: {out / "submission.csv"} ({len(ids)} rows)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'validate', 'refit', 'predict'])
    parser.add_argument('--zip', type=Path)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--template', type=Path)
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, default=300)
    parser.add_argument('--architecture', choices=['efficientnet_b0', 'efficientnet_b3'], default='efficientnet_b3')
    parser.add_argument('--tta', action=argparse.BooleanOptionalAction, default=True,
                        help='Average eight views in validation and prediction')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--ensemble-seeds', type=int, nargs='+', default=[2026, 3407])
    parser.add_argument('--checkpoints', type=Path, nargs='+')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Keep downloaded weights inside the chosen output workspace.
    torch.hub.set_dir(str(args.out / '.torch_cache'))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if args.zip:
        unpack(args.zip, args.data)
    root = locate_data(args.data)
    template = args.template or locate_template(args.data)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    classes, records = collect(root)
    print(f'Device: {device}, classes: {classes}', flush=True)
    if args.mode != 'predict':
        save_json(args.out / 'environment.json', {'torch': torch.__version__,
            'torchvision': torchvision.__version__, 'numpy': np.__version__,
            'device': str(device), 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
            'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    if args.mode in ('run', 'validate'):
        train, valid = stratified_split(records, args.seed)
        save_json(args.out / 'split.json', {'seed': args.seed,
            'train': [str(p.relative_to(root)) for p, _ in train],
            'valid': [str(p.relative_to(root)) for p, _ in valid]})
        selected_epoch = fit(train, valid, classes, args, args.out / 'validation', device)
        if args.mode == 'validate':
            return
        args.epochs = selected_epoch
    if args.mode in ('run', 'refit'):
        paths = []
        for seed in args.ensemble_seeds:
            args.seed = seed
            destination = args.out / f'full_seed_{seed}'
            print(f'FULL REFIT seed={seed} epochs={args.epochs}', flush=True)
            fit(records, None, classes, args, destination, device)
            paths.append(destination / 'model.pt')
        predict(root, template, paths, args.out, device, args.batch_size)
    elif args.mode == 'predict':
        if not args.checkpoints:
            parser.error('predict requires --checkpoints')
        predict(root, template, args.checkpoints, args.out, device, args.batch_size)

if __name__ == "__main__":
    main()
