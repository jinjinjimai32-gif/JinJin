import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASSES = ("Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default="/kaggle/working" if Path("/kaggle").exists() else "emotion_results")
    parser.add_argument("--classes", default=",".join(CLASSES))
    parser.add_argument("--epochs-cnn", type=int, default=70)
    parser.add_argument("--epochs-transfer", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-ratio", type=float, default=0.12)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--class-balance", type=float, default=0.0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--sample-submission", default=None)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignored notebook arguments:", unknown)
    if args.batch_size < 2 or args.workers < 0 or args.patience < 1:
        parser.error("batch-size >= 2, workers >= 0, patience >= 1 required")
    if args.epochs_cnn < 1 or args.epochs_transfer < 1:
        parser.error("epoch counts must be positive")
    if not 0 < args.val_ratio < 0.5 or not 0 <= args.class_balance <= 1:
        parser.error("val-ratio must be in (0, 0.5); class-balance in [0, 1]")
    return args


def locate_data(value):
    roots = [Path(value).expanduser()] if value else [Path("/kaggle/input"), Path.cwd()]
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".zip":
            return root
        if not root.is_dir():
            continue
        if (root / "train").is_dir() and (root / "test").is_dir():
            return root
        for candidate in sorted(root.rglob("train")):
            if candidate.is_dir() and (candidate.parent / "test").is_dir():
                return candidate.parent
        for candidate in sorted(root.rglob("*.zip")):
            with zipfile.ZipFile(candidate) as archive:
                names = archive.namelist()
                if any("/train/" in "/" + n for n in names) and any("/test/" in "/" + n for n in names):
                    return candidate
    raise FileNotFoundError("Data not found. Pass --data with the dataset folder or ZIP file.")


def load_images(source, classes, sample_path=None):
    archive = zipfile.ZipFile(source) if source.is_file() else None
    entries = sorted(archive.namelist()) if archive else sorted(p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file())
    train_images, labels, train_names, test_images, test_names = [], [], [], [], []
    mapping = {name.lower(): i for i, name in enumerate(classes)}
    template = None
    def read_bytes(name):
        return archive.read(name) if archive else (source / name).read_bytes()
    try:
        for name in entries:
            parts = Path(name).parts
            if Path(name).name.lower() in {"submission.csv", "sample_submission.csv"}:
                candidate = pd.read_csv(io.BytesIO(read_bytes(name)))
                if list(candidate.columns) == ["ID", "Emotion"]:
                    template = candidate
            if Path(name).suffix.lower() not in EXTENSIONS:
                continue
            lower = [p.lower() for p in parts]
            is_train = "train" in lower
            is_test = "test" in lower
            if not is_train and not is_test:
                continue
            with Image.open(io.BytesIO(read_bytes(name))) as image:
                array = np.asarray(image.convert("L").resize((48, 48), Image.Resampling.BILINEAR)).copy()
            if is_train:
                label_name = parts[-2].lower()
                if label_name not in mapping:
                    raise ValueError(f"Unrecognized training class: {parts[-2]}")
                train_images.append(array)
                labels.append(mapping[label_name])
                train_names.append(name)
            else:
                test_images.append(array)
                test_names.append(Path(name).name)
        if sample_path:
            template = pd.read_csv(sample_path)
        elif template is None and source.is_dir():
            for parent in (source.parent, source.parent.parent):
                found = False
                for filename in ("sample_submission.csv", "submission.csv"):
                    path = parent / filename
                    if path.is_file():
                        candidate = pd.read_csv(path)
                        if list(candidate.columns) == ["ID", "Emotion"] and len(candidate) == len(test_names):
                            template = candidate
                            found = True
                            break
                if found:
                    break
    finally:
        if archive:
            archive.close()
    if not train_images or not test_images:
        raise ValueError("Both labeled training images and test images are required.")
    if len(set(test_names)) != len(test_names):
        raise ValueError("Duplicate test filenames detected.")
    y = np.asarray(labels, dtype=np.int64)
    if np.any(np.bincount(y, minlength=len(classes)) < 3):
        raise ValueError("Each class must contain at least three training examples.")
    return np.stack(train_images), y, train_names, np.stack(test_images), test_names, template


def grouped_split(images, labels, ratio, seed):
    rng = np.random.default_rng(seed)
    groups = {}
    for i, image in enumerate(images):
        digest = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
        groups.setdefault(digest, []).append(i)
    buckets = {}
    for indices in groups.values():
        majority = int(np.bincount(labels[indices]).argmax())
        buckets.setdefault(majority, []).append(indices)
    train, valid = [], []
    for label, members in sorted(buckets.items()):
        rng.shuffle(members)
        target = max(1, round(sum(len(g) for g in members) * ratio))
        count = 0
        for group in members:
            if count < target and len(members) > 1:
                valid.extend(group)
                count += len(group)
            else:
                train.extend(group)
    train, valid = np.asarray(sorted(train)), np.asarray(sorted(valid))
    if len(train) < 2 or len(valid) < 2 or set(labels[train]) != set(labels) or set(labels[valid]) != set(labels):
        raise ValueError("Unable to create a valid grouped class split.")
    return train, valid


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Faces(Dataset):
    def __init__(self, images, labels, indices, size, rgb, training, mean, std):
        self.images, self.labels, self.indices = images, labels, indices
        self.rgb = rgb
        steps = [transforms.Resize((size, size), antialias=True)]
        if training:
            steps += [transforms.RandomHorizontalFlip(), transforms.RandomAffine(10, translate=(0.07, 0.07), scale=(0.9, 1.1), fill=128)]
            steps += [transforms.ColorJitter(brightness=0.12, contrast=0.15)]
        steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        if training:
            steps += [transforms.RandomErasing(p=0.2, scale=(0.015, 0.08), value=0)]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        image = Image.fromarray(self.images[i])
        if self.rgb:
            image = image.convert("RGB")
        return self.transform(image), int(self.labels[i]) if self.labels is not None else i


class ChannelGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, max(8, channels // 16), 1), nn.SiLU(), nn.Conv2d(max(8, channels // 16), channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.layers(x)


class ResidualUnit(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.path = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), ChannelGate(cout))
        self.shortcut = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.silu(self.path(x) + self.shortcut(x))


class ExpressionNet(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 48, 3, padding=1, bias=False), nn.BatchNorm2d(48), nn.SiLU(), ResidualUnit(48, 64), ResidualUnit(64, 96, 2), ResidualUnit(96, 96), ResidualUnit(96, 192, 2), nn.Dropout2d(0.1), ResidualUnit(192, 192), ResidualUnit(192, 256, 2), ResidualUnit(256, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, count))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(kind, count, pretrained):
    if kind == "attention_cnn":
        return ExpressionNet(count)
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    try:
        model = models.resnet18(weights=weights)
    except (OSError, RuntimeError) as error:
        raise RuntimeError("Pretrained weights could not be loaded. Enable Kaggle Internet, cache the weights, or use --no-pretrained explicitly.") from error
    model.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(model.fc.in_features, count))
    return model


def loader(dataset, args, shuffle=False):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0, drop_last=shuffle and len(dataset) % args.batch_size == 1)


@torch.inference_mode()
def probabilities(model, batches, device):
    model.eval()
    results = []
    for x, _ in batches:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            original = model(x).float().softmax(1)
            flipped = model(x.flip(-1)).float().softmax(1)
        results.append(((original + flipped) * 0.5).cpu().numpy())
    return np.concatenate(results)


def update_average(average, model, step):
    decay = min(0.997, (1 + step) / (10 + step))
    with torch.no_grad():
        for target, current in zip(average.parameters(), model.parameters()):
            target.lerp_(current.detach(), 1 - decay)
        for target, current in zip(average.buffers(), model.buffers()):
            target.copy_(current)


def fit(model, train_batches, valid_batches, targets, epochs, kind, args, device, metadata):
    if kind == "resnet18":
        groups = [{"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")], "lr": 0.00012}, {"params": model.fc.parameters(), "lr": 0.0006}]
    else:
        groups = [{"params": model.parameters(), "lr": 0.0008}]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0003)
    total_steps = epochs * len(train_batches)
    warmup = max(1, int(total_steps * 0.05))
    def schedule(step):
        if step < warmup:
            return 0.1 + 0.9 * step / warmup
        fraction = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.01 + 0.99 * (1 + math.cos(math.pi * fraction)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    counts = np.bincount(train_batches.dataset.labels[train_batches.dataset.indices], minlength=len(metadata["classes"]))
    weight = np.power(counts.mean() / np.maximum(counts, 1), args.class_balance)
    weight = torch.tensor(weight / weight.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
    average = copy.deepcopy(model).eval()
    average.requires_grad_(False)
    best_key, best_epoch, step, records = (-1.0, -float("inf")), 0, 0, []
    checkpoint = Path(args.output) / f"{kind}_best.pt"
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        running, seen = 0.0, 0
        for x, y in train_batches:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
                step += 1
                update_average(average, model, step)
            running += loss.item() * y.size(0)
            seen += y.size(0)
        prediction = probabilities(average, valid_batches, device)
        accuracy = float((prediction.argmax(1) == targets).mean())
        nll = float(-np.log(prediction[np.arange(len(targets)), targets].clip(1e-8)).mean())
        record = {"epoch": epoch, "train_loss": running / seen, "val_accuracy": accuracy, "val_nll": nll, "seconds": round(time.time() - start, 2)}
        records.append(record)
        if (accuracy, -nll) > best_key:
            best_key, best_epoch = (accuracy, -nll), epoch
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in average.state_dict().items()}, "metadata": metadata, "epoch": epoch, "val_accuracy": accuracy}, checkpoint)
        pd.DataFrame(records).to_csv(Path(args.output) / f"{kind}_history.csv", index=False)
        print(f"{kind} {epoch}/{epochs} loss={running / seen:.4f} val={accuracy:.4f} best={best_key[0]:.4f} time={record['seconds']:.0f}s", flush=True)
        if epoch - best_epoch >= args.patience:
            break
    del average
    return checkpoint


def write_submission(names, probabilities_array, template, path):
    if len(names) != len(probabilities_array) or not np.isfinite(probabilities_array).all():
        raise ValueError("Invalid predictions.")
    result = pd.DataFrame({"ID": names, "Emotion": probabilities_array.argmax(1).astype(np.int64)})
    if template is not None:
        if list(template.columns) != ["ID", "Emotion"] or template["ID"].duplicated().any() or set(template["ID"]) != set(names):
            raise ValueError("Sample submission IDs or columns do not match the test data.")
        result = result.set_index("ID").loc[template["ID"].tolist()].reset_index()
    if result["Emotion"].min() < 0 or result["Emotion"].max() >= probabilities_array.shape[1]:
        raise ValueError("Predicted labels are out of range.")
    result.to_csv(path, index=False)
    return result


def main():
    args = arguments()
    classes = tuple(s.strip() for s in args.classes.split(","))
    if len(classes) != 6 or len(set(s.lower() for s in classes)) != 6:
        raise ValueError("Exactly six unique class names are required in numeric label order.")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = locate_data(args.data)
    images, labels, train_names, tests, names, template = load_images(source, classes, args.sample_submission)
    train_ids, valid_ids = grouped_split(images, labels, args.val_ratio, args.seed)
    if args.smoke_test:
        train_ids = np.concatenate([train_ids[labels[train_ids] == i][:4] for i in range(6)])
        valid_ids = np.concatenate([valid_ids[labels[valid_ids] == i][:2] for i in range(6)])
        tests, names, template = tests[:12], names[:12], None
        args.epochs_cnn = args.epochs_transfer = 1
        args.batch_size = min(12, args.batch_size)
        args.no_pretrained = True
    mean = float(images[train_ids].mean() / 255)
    std = max(float(images[train_ids].std() / 255), 0.001)
    split_digest = hashlib.sha256(json.dumps({"train": [train_names[i] for i in train_ids], "valid": [train_names[i] for i in valid_ids]}, sort_keys=True).encode()).hexdigest()
    print("Device:", device, "Data:", source, flush=True)
    print("Classes:", dict(enumerate(classes)), "Train:", len(train_ids), "Validation:", len(valid_ids), "Test:", len(names), flush=True)
    pd.DataFrame({"file": train_names, "label": labels, "split": np.where(np.isin(np.arange(len(labels)), valid_ids), "validation", np.where(np.isin(np.arange(len(labels)), train_ids), "train", "unused"))}).to_csv(out / "split.csv", index=False)
    validation_predictions, test_predictions, model_scores = [], [], {}
    settings = [("attention_cnn", 48, False, args.epochs_cnn), ("resnet18", 112, True, args.epochs_transfer)]
    for offset, (kind, size, rgb, epochs) in enumerate(settings):
        seed_all(args.seed + offset * 97)
        normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if rgb else ([mean], [std])
        metadata = {"classes": list(classes), "kind": kind, "size": size, "mean": normalization[0], "std": normalization[1], "split_digest": split_digest}
        train_set = Faces(images, labels, train_ids, size, rgb, True, *normalization)
        valid_set = Faces(images, labels, valid_ids, size, rgb, False, *normalization)
        test_set = Faces(tests, None, np.arange(len(tests)), size, rgb, False, *normalization)
        valid_batches = loader(valid_set, args)
        checkpoint = out / f"{kind}_best.pt"
        model = build_model(kind, len(classes), pretrained=rgb and not args.no_pretrained and not args.predict_only).to(device)
        if not args.predict_only:
            fit(model, loader(train_set, args, True), valid_batches, labels[valid_ids], epochs, kind, args, device, metadata)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["metadata"] != metadata:
            raise ValueError(f"Checkpoint settings or validation split differ: {checkpoint}")
        model.load_state_dict(saved["state_dict"])
        del saved
        val_probs = probabilities(model, valid_batches, device)
        test_probs = probabilities(model, loader(test_set, args), device)
        validation_predictions.append(val_probs)
        test_predictions.append(test_probs)
        model_scores[kind] = float((val_probs.argmax(1) == labels[valid_ids]).mean())
        np.savez_compressed(out / f"{kind}_probabilities.npz", validation=val_probs, test=test_probs, validation_indices=valid_ids, test_ids=np.asarray(names))
        del model, train_set, valid_set, test_set, valid_batches
        if device.type == "cuda":
            torch.cuda.empty_cache()
    targets = labels[valid_ids]
    choices = []
    for alpha in np.linspace(0, 1, 21):
        combined = alpha * validation_predictions[0] + (1 - alpha) * validation_predictions[1]
        accuracy = float((combined.argmax(1) == targets).mean())
        nll = float(-np.log(combined[np.arange(len(targets)), targets].clip(1e-8)).mean())
        choices.append((accuracy, -nll, -abs(alpha - 0.5), float(alpha)))
    best_accuracy, negative_nll, _, alpha = max(choices)
    final_probs = alpha * test_predictions[0] + (1 - alpha) * test_predictions[1]
    combined_validation = alpha * validation_predictions[0] + (1 - alpha) * validation_predictions[1]
    matrix = np.zeros((6, 6), dtype=np.int64)
    np.add.at(matrix, (targets, combined_validation.argmax(1)), 1)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(out / "validation_confusion.csv")
    result = write_submission(names, final_probs, template, out / "submission.csv")
    report = {"classes": dict(enumerate(classes)), "training_count": len(train_ids), "validation_count": len(valid_ids), "test_count": len(names), "model_accuracy": model_scores, "fusion_validation_accuracy": best_accuracy, "fusion_validation_nll": -negative_nll, "attention_cnn_weight": alpha, "resnet18_weight": 1 - alpha, "seed": args.seed, "smoke_test": args.smoke_test, "pretrained_requested": not args.no_pretrained, "validation_used_for_model_selection": True, "per_class_recall": {name: float(matrix[i, i] / max(1, matrix[i].sum())) for i, name in enumerate(classes)}}
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print("Submission:", (out / "submission.csv").resolve(), "Rows:", len(result), flush=True)
    if args.smoke_test:
        print("SMOKE TEST ONLY: this submission contains a subset and must not be submitted.")


if __name__ == "__main__":
    main()