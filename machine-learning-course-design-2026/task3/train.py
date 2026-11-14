from __future__ import annotations
import argparse
import gc
import copy
import hashlib
import json
import math
import os
import sys
import random
import re
import time
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights, resnet34
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FUNDUS_CROP = True
EXPORT_SUBMISSION = True

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', default='', help='detection 目录；留空自动扫描 /kaggle/input')
    p.add_argument('--output-dir', default='/kaggle/working/task3_precision_v4')
    p.add_argument('--weights', default='', help='可选：本地 torchvision ResNet34 的 .pth 权重')
    p.add_argument('--pretrained', choices=['auto', 'require', 'none'], default='auto')
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--min-epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--sigma', type=float, default=2.0, help='热图高斯标准差，以热图像素计')
    p.add_argument('--no-tta', action='store_true')
    p.add_argument('--resume', action='store_true', help='复用同配置已完成的折；不恢复中断的 epoch')
    p.add_argument('--smoke', action='store_true', help='128 分辨率、每折 1 轮，只检查流程')
    p.add_argument('--refine-size', type=int, default=384)
    p.add_argument('--refine-epochs', type=int, default=70)
    p.add_argument('--refine-patience', type=int, default=20)
    p.add_argument('--refine-min-epochs', type=int, default=25)
    p.add_argument('--roi-size', type=float, default=768.0)
    p.add_argument('--refine-repeat', type=int, default=2)
    p.add_argument('--refine-lr', type=float, default=0.0004)
    p.add_argument('--refine-sigma', type=float, default=1.5)
    p.add_argument('--refine-ema', type=float, default=0.97)
    p.add_argument('--coarse-checkpoints', default='')
    p.add_argument('--experiment-fold', type=int, default=-1)
    p.add_argument('--no-image-cache', action='store_true')
    p.add_argument('--no-fundus-crop', action='store_true')
    p.add_argument('--model-source', default='auto')
    p.add_argument('--split-seed', type=int, default=2026)
    p.add_argument('--alternative-seed', type=int, default=3407)
    p.add_argument('--alternative-size', type=int, default=640)
    p.add_argument('--alternative-refine-size', type=int, default=512)
    p.add_argument('--baseline-only', action='store_true')
    if argv is None and 'ipykernel' in sys.modules:
        argv = []
    a = p.parse_args(argv)
    if a.smoke:
        a.image_size, a.epochs, a.patience, a.min_epochs, a.workers = (128, 1, 1, 1, 0)
        a.refine_size, a.refine_epochs, a.refine_patience, a.refine_min_epochs, a.refine_repeat = (128, 2, 2, 1, 1)
        a.alternative_size, a.alternative_refine_size = 160, 160
    if a.image_size < 64 or a.image_size % 32:
        p.error('--image-size 必须是至少 64 的 32 倍数')
    if a.folds < 2 or a.epochs < 1 or a.batch_size < 1 or (a.patience < 1) or (a.sigma <= 0):
        p.error('folds>=2, epochs>=1, batch-size>=1, patience>=1, sigma>0')
    if a.refine_size < 64 or a.refine_size % 32 or a.refine_epochs < 1 or (a.refine_repeat < 1) or (a.roi_size <= 0) or (a.refine_patience < 1) or (a.refine_lr <= 0) or (a.refine_sigma <= 0) or (not 0 <= a.refine_ema < 1):
        p.error('Invalid refinement configuration')
    if not -1 <= a.experiment_fold < a.folds:
        p.error('experiment-fold must be -1 or a valid fold index')
    if any(size < 64 or size % 32 for size in [a.alternative_size, a.alternative_refine_size]):
        p.error('Alternative resolutions must be multiples of 32 and at least 64.')
    return a

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def worker_seed(worker_id):
    s = torch.initial_seed() % 2 ** 32
    random.seed(s)
    np.random.seed(s)
    cv2.setNumThreads(0)

def normalize_id(value):
    s = Path(str(value).strip().replace('\\', '/')).stem
    if re.fullmatch('\\d+(?:\\.0+)?', s):
        return str(int(float(s)))
    return s.casefold()

def safe_extract(archive, target):
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = member.filename.replace('\\', '/')
            dest = (target / name).resolve()
            if not dest.is_relative_to(target):
                raise ValueError(f'压缩包路径越界: {name}')
            if member.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, dest.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)

def find_data(args, out):
    roots = [Path(args.data_root)] if args.data_root else [Path('/kaggle/input')]
    candidates = []
    for base in roots:
        if base.is_file() and base.suffix.lower() == '.zip':
            extracted = out / 'extracted_data'
            safe_extract(base, extracted)
            base = extracted
        if not base.exists():
            continue
        candidates += list(base.rglob('fovea_localization_train_GT.csv'))
    if not candidates and (not args.data_root) and Path('/kaggle/input').exists():
        for archive in Path('/kaggle/input').rglob('*.zip'):
            with zipfile.ZipFile(archive) as z:
                if any((n.endswith('fovea_localization_train_GT.csv') for n in z.namelist())):
                    folder = out / ('extracted_' + hashlib.sha256(str(archive).encode()).hexdigest()[:10])
                    safe_extract(archive, folder)
                    candidates += list(folder.rglob('fovea_localization_train_GT.csv'))
    candidates = sorted({p.resolve() for p in candidates})
    if len(candidates) > 1:
        competition_candidates = [p for p in candidates if '/competitions/' in p.as_posix()]
        if len(competition_candidates) == 1:
            candidates = competition_candidates
    if len(candidates) != 1:
        raise RuntimeError(f'找到 {len(candidates)} 份训练 CSV。请用 --data-root 指向唯一的 detection 目录或 zip。\n{candidates}')
    root = candidates[0].parent
    for name in ['train', 'test', 'sample_submission.csv']:
        if not (root / name).exists():
            raise FileNotFoundError(root / name)
    return root

def image_index(folder):
    found = {}
    for p in sorted(folder.rglob('*')):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            key = normalize_id(p.stem)
            if key in found:
                raise ValueError(f'重复图片编号: {found[key]}, {p}')
            found[key] = p
    return found

def read_rgb(path):
    data = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if data is None:
        raise ValueError(f'读取图片失败: {path}')
    return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

def fundus_bounds(image):
    h, w = image.shape[:2]
    scale = min(1.0, 768 / max(h, w))
    sw, sh = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    count, _, stats, _ = cv2.connectedComponentsWithStats((gray > 10).astype(np.uint8), connectivity=8)
    if count < 2:
        return (0, 0, w, h)
    x, y, bw, bh, area = stats[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])]
    if area < 0.08 * sw * sh or bw < 0.35 * sw or bh < 0.35 * sh:
        return (0, 0, w, h)
    margin = 0.025 * max(bw, bh)
    x0, y0 = (max(0, math.floor((x - margin) * w / sw)), max(0, math.floor((y - margin) * h / sh)))
    x1, y1 = (min(w, math.ceil((x + bw + margin) * w / sw)), min(h, math.ceil((y + bh + margin) * h / sh)))
    return (x0, y0, x1, y1)

def letterbox(image, size):
    h, w = image.shape[:2]
    x0, y0, x1, y1 = fundus_bounds(image) if FUNDUS_CROP else (0, 0, w, h)
    crop = image[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    ratio = size / max(ch, cw)
    nw, nh = (max(1, round(cw * ratio)), max(1, round(ch * ratio)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    left, top = ((size - nw) // 2, (size - nh) // 2)
    result = np.zeros((size, size, 3), dtype=np.uint8)
    result[top:top + nh, left:left + nw] = resized
    sx, sy = (nw / cw, nh / ch)
    return (result, np.array([sx, sy, left - x0 * sx, top - y0 * sy, w, h], np.float32))

def to_model_xy(xy, meta, size):
    return ((np.asarray(xy) + 0.5) * meta[:2] - 0.5 + meta[2:4]) / (size - 1)

def to_original_xy(xy, meta, size):
    return (np.asarray(xy) * (size - 1) - meta[..., 2:4] + 0.5) / meta[..., :2] - 0.5

def read_records(root, size):
    df = pd.read_csv(root / 'fovea_localization_train_GT.csv', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    required = ['data', 'Fovea_X', 'Fovea_Y']
    if not set(required).issubset(df.columns):
        raise ValueError(f'训练 CSV 缺少列 {required}: {list(df.columns)}')
    train_files, test_files = (image_index(root / 'train'), image_index(root / 'test'))
    records, seen = ([], set())
    for row in df[required].itertuples(index=False, name=None):
        key, xy = (normalize_id(row[0]), np.array(row[1:], dtype=np.float32))
        if key in seen or key not in train_files or (not np.isfinite(xy).all()):
            raise ValueError(f'训练标注重复、缺图或坐标无效: {row}')
        seen.add(key)
        image, meta = letterbox(read_rgb(train_files[key]), size)
        visible = not np.all(xy == 0)
        if visible and (not (0 <= xy[0] < meta[4] and 0 <= xy[1] < meta[5])):
            raise ValueError(f'训练坐标越界: {row}, image size={meta[4:]}')
        records.append(dict(key=key, path=train_files[key], image=image, meta=meta, xy=xy, point=to_model_xy(xy, meta, size) if visible else np.zeros(2, np.float32), visible=float(visible)))
    sample = pd.read_csv(root / 'sample_submission.csv', encoding='utf-8-sig', dtype={'ImageID': str})
    if list(sample.columns) != ['ImageID', 'value'] or sample.ImageID.duplicated().any():
        raise ValueError('sample_submission 必须包含唯一 ImageID 和 value 两列')
    sample_keys, axes = ([], defaultdict(set))
    for item in sample.ImageID:
        match = re.fullmatch('(.+)_Fovea_([XY])', str(item))
        if not match:
            raise ValueError(f'未知的提交编号格式: {item}')
        key, axis = (normalize_id(match[1]), match[2])
        if key not in test_files:
            raise ValueError(f'提交编号找不到测试图片: {item}')
        if key not in sample_keys:
            sample_keys.append(key)
        axes[key].add(axis)
    if any((v != {'X', 'Y'} for v in axes.values())):
        raise ValueError('每张测试图必须各有一行 X 和 Y')
    tests = []
    for key in sample_keys:
        image, meta = letterbox(read_rgb(test_files[key]), size)
        tests.append(dict(key=key, path=test_files[key], image=image, meta=meta, xy=np.zeros(2, np.float32), point=np.zeros(2, np.float32), visible=0.0))
    print(f'Data: {root}\nTrain={len(records)}, Test={len(tests)}, Submission rows={len(sample)}')
    return (records, tests, sample)

def make_folds(records, count, seed):
    if count > len(records):
        raise ValueError('折数不能超过训练样本数')
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r['visible'], int(r['meta'][4]), int(r['meta'][5])].append(i)
    rng, folds = (np.random.default_rng(seed), [[] for _ in range(count)])
    for key in sorted(groups):
        ids = np.array(groups[key])
        rng.shuffle(ids)
        order = sorted(range(count), key=lambda f: (len(folds[f]), f))
        for j, idx in enumerate(ids):
            folds[order[j % count]].append(int(idx))
    return [np.array(sorted(ids), dtype=np.int64) for ids in folds]

class FoveaDataset(Dataset):

    def __init__(self, records, augment=False):
        self.records, self.augment = (records, augment)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        image, point = (r['image'].copy(), r['point'].copy())
        size = image.shape[0]
        if self.augment:
            if random.random() < 0.5:
                image = image[:, ::-1].copy()
                if r['visible']:
                    point[0] = 1 - point[0]
            for _ in range(8):
                matrix = cv2.getRotationMatrix2D(((size - 1) / 2, (size - 1) / 2), random.uniform(-15, 15), random.uniform(0.9, 1.1))
                matrix[:, 2] += np.random.uniform(-0.04, 0.04, size=2) * size
                transformed = matrix @ np.r_[point * (size - 1), 1.0]
                if not r['visible'] or ((transformed >= 1) & (transformed <= size - 2)).all():
                    image = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR)
                    if r['visible']:
                        point = (transformed / (size - 1)).astype(np.float32)
                    break
            image = image.astype(np.float32) / 255.0
            if random.random() < 0.8:
                image = np.clip(image * random.uniform(0.8, 1.2), 0, 1)
                image = image ** random.uniform(0.8, 1.2)
                gray = image.mean(axis=2, keepdims=True)
                image = np.clip(gray + random.uniform(0.8, 1.2) * (image - gray), 0, 1)
            if random.random() < 0.15:
                image = np.clip(image + np.random.normal(0, 0.008, image.shape).astype(np.float32), 0, 1)
        else:
            image = image.astype(np.float32) / 255.0
        image = ((image - MEAN) / STD).transpose(2, 0, 1)
        return dict(image=torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)), point=torch.tensor(point, dtype=torch.float32), visible=torch.tensor(r['visible'], dtype=torch.float32), meta=torch.tensor(r['meta']), xy=torch.tensor(r['xy']))

def make_loader(records, args, train=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(FoveaDataset(records, augment=train), batch_size=args.batch_size, shuffle=train, num_workers=args.workers, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_seed, generator=generator, drop_last=False)

def get_encoder_state(args):
    if args.pretrained == 'none':
        return (None, 'random initialization')
    weights_path = Path(args.weights) if args.weights else None
    if weights_path is None and Path('/kaggle/input').exists():
        candidates = sorted(Path('/kaggle/input').rglob('resnet34*.pth'))
        if candidates:
            weights_path = candidates[0]
    if weights_path is not None:
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        return (state, f'local torchvision weights: {weights_path}')
    try:
        encoder = resnet34(weights=ResNet34_Weights.DEFAULT)
        return (encoder.state_dict(), 'torchvision ImageNet1K pretrained ResNet34')
    except Exception as exc:
        if args.pretrained == 'require':
            raise RuntimeError('无法下载预训练权重。启用 Internet，或通过 --weights 提供 resnet34 .pth。') from exc
        warnings.warn(f'预训练权重不可用，改为随机初始化，效果可能明显下降。错误: {exc}')
        return (None, 'random initialization (pretrained download unavailable)')

class ConvBlock(nn.Sequential):

    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(True), nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(True))

class UpBlock(nn.Module):

    def __init__(self, cin, skip, cout):
        super().__init__()
        self.block = ConvBlock(cin + skip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))

class FoveaNet(nn.Module):

    def __init__(self, encoder_state=None):
        super().__init__()
        self.encoder = resnet34(weights=None)
        if encoder_state is not None:
            self.encoder.load_state_dict(encoder_state, strict=True)
        self.encoder.fc = nn.Identity()
        self.up3 = UpBlock(512, 256, 192)
        self.up2 = UpBlock(192, 128, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.heat_head = nn.Conv2d(64, 1, 1)
        self.visible_head = nn.Linear(512, 1)

    def forward(self, x):
        e = self.encoder
        x = e.maxpool(e.relu(e.bn1(e.conv1(x))))
        a = e.layer1(x)
        b = e.layer2(a)
        c = e.layer3(b)
        d = e.layer4(c)
        decoded = self.up1(self.up2(self.up3(d, c), b), a)
        return (self.heat_head(decoded), self.visible_head(d.mean(dim=(2, 3))).squeeze(1))

def heatmap_points(logits):
    logits = logits.float()
    b, _, h, w = logits.shape
    prob = logits.flatten(1).softmax(1).reshape(b, h, w)
    xs = torch.linspace(0, 1, w, device=logits.device)
    ys = torch.linspace(0, 1, h, device=logits.device)
    return torch.stack([(prob.sum(1) * xs).sum(1), (prob.sum(2) * ys).sum(1)], 1)

def training_loss(logits, vis_logits, point, visible, use_visibility, sigma):
    logits = logits.float()
    b, _, h, w = logits.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=logits.device), torch.arange(w, device=logits.device), indexing='ij')
    gx, gy = (point[:, 0, None, None] * (w - 1), point[:, 1, None, None] * (h - 1))
    target = torch.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sigma ** 2))
    target = target / target.sum((1, 2), keepdim=True).clamp_min(1e-12)
    ce = -(target.flatten(1) * logits.flatten(1).log_softmax(1)).sum(1)
    coord = ((heatmap_points(logits) - point) ** 2).mean(1)
    loc = ((ce + 100 * coord) * visible).sum() / visible.sum().clamp_min(1.0)
    if use_visibility:
        loc = loc + F.binary_cross_entropy_with_logits(vis_logits.float(), visible)
    return loc

@torch.no_grad()
def predict(model, loader, device, args, use_visibility):
    model.eval()
    predictions, probabilities = ([], [])
    for batch in loader:
        image = batch['image'].to(device, non_blocking=True)
        variants = [False] if args.no_tta else [False, True]
        xy_all, vis_all = ([], [])
        for flip in variants:
            inp = image.flip(-1) if flip else image
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                heat, vis = model(inp)
            xy = heatmap_points(heat)
            if flip:
                xy[:, 0] = 1 - xy[:, 0]
            xy_all.append(xy.cpu().numpy())
            vis_all.append(vis.float().sigmoid().cpu().numpy())
        normalized = np.mean(xy_all, axis=0)
        meta = batch['meta'].numpy()
        restored = to_original_xy(normalized, meta, args.image_size)
        restored = np.maximum(0, np.minimum(restored, meta[:, 4:6] - 1))
        predictions.append(restored)
        probabilities.append(np.mean(vis_all, axis=0) if use_visibility else np.ones(len(image)))
    return (np.concatenate(predictions), np.concatenate(probabilities))

def rmse(pred, truth):
    return float(np.sqrt(np.mean((np.asarray(pred, np.float64) - np.asarray(truth, np.float64)) ** 2)))

def apply_visibility(xy, prob, threshold, use_visibility):
    result = xy.copy()
    if use_visibility:
        result[prob < threshold] = 0
    return result

def dataset_signature(records, tests):
    digest = hashlib.sha256()
    for r in records + tests:
        digest.update(r['key'].encode())
        digest.update(r['xy'].tobytes())
        digest.update(r['path'].read_bytes())
    return digest.hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def train_fold(fold, train_records, val_records, encoder_state, args, device, use_visibility, out, signature):
    seed_all(args.seed + fold)
    checkpoint = out / f'fold_{fold}.pth'
    marker = out / f'fold_{fold}_complete.json'
    if args.resume and marker.exists() and checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['signature'] != signature:
            raise RuntimeError(f'fold {fold} 与当前数据/配置不一致。请更换输出目录或取消 --resume。')
        model = FoveaNet().to(device)
        model.load_state_dict(saved['model'])
        pred, prob = predict(model, make_loader(val_records, args), device, args, use_visibility)
        del model
        return (pred, prob)
    if marker.exists():
        marker.unlink()
    model = FoveaNet(encoder_state).to(device)
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    other = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': args.lr * 0.1}, {'params': other, 'lr': args.lr}], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.001)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    train_loader = make_loader(train_records, args, True, args.seed + fold)
    val_loader = make_loader(val_records, args)
    truth = np.stack([r['xy'] for r in val_records])
    best, bad_epochs, history = (float('inf'), 0, [])
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        if encoder_state is not None:
            for module in model.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        loss_sum, seen = (0.0, 0)
        for batch in train_loader:
            image = batch['image'].to(device, non_blocking=True)
            point = batch['point'].to(device, non_blocking=True)
            visible = batch['visible'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits, vis_logits = model(image)
                loss = training_loss(logits, vis_logits, point, visible, use_visibility, args.sigma)
            if not torch.isfinite(loss):
                raise RuntimeError(f'fold={fold}, epoch={epoch}: non-finite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(image)
            seen += len(image)
        scheduler.step()
        val_xy, val_prob = predict(model, val_loader, device, args, use_visibility)
        val_pred = apply_visibility(val_xy, val_prob, 0.5, use_visibility)
        score = rmse(val_pred, truth)
        history.append(dict(epoch=epoch, train_loss=loss_sum / seen, val_rmse=score))
        if score < best:
            best, bad_epochs = (score, 0)
            torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, epoch=epoch, val_rmse=score, signature=signature, image_size=args.image_size, use_visibility=use_visibility), checkpoint)
        else:
            bad_epochs += 1
        print(f'Fold {fold + 1}/{args.folds} Epoch {epoch:03d} loss={loss_sum / seen:.4f} val_RMSE={score:.3f}px best={best:.3f}px time={time.time() - started:.1f}s', flush=True)
        if epoch >= args.min_epochs and bad_epochs >= args.patience:
            break
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'])
    pred, prob = predict(model, val_loader, device, args, use_visibility)
    pd.DataFrame(history).to_csv(out / f'fold_{fold}_history.csv', index=False)
    write_json(marker, dict(best_epoch=saved['epoch'], best_rmse=best, signature=signature))
    del model, optimizer, scheduler, train_loader, val_loader, scaler, saved
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return (pred, prob)

def save_submission(sample, tests, pred, out):
    mapping = {r['key']: xy for r, xy in zip(tests, pred)}
    submission = sample.copy()
    values = []
    for item in submission.ImageID:
        stem, axis = item.rsplit('_Fovea_', 1)
        values.append(float(mapping[normalize_id(stem)][0 if axis == 'X' else 1]))
    submission['value'] = values
    if not np.isfinite(submission.value).all() or (submission.value < 0).any():
        raise ValueError('提交包含无效坐标')
    if not submission.ImageID.equals(sample.ImageID):
        raise AssertionError('提交的 ImageID 顺序发生变化')
    path = out / 'submission.csv'
    submission.to_csv(path, index=False, float_format='%.6f')
    check = pd.read_csv(path, dtype={'ImageID': str})
    if len(check) != len(sample) or list(check.columns) != ['ImageID', 'value'] or check.isna().any().any():
        raise AssertionError('提交文件自检失败')
    kaggle_root = Path('/kaggle/working')
    if kaggle_root.is_dir() and EXPORT_SUBMISSION:
        root_path = kaggle_root / 'submission.csv'
        if root_path.resolve() != path.resolve():
            submission.to_csv(root_path, index=False, float_format='%.6f')
        print(f'Submission filename: submission.csv; saved: {root_path}')
    print(f'\nSubmission saved: {path}\n{submission.head(8).to_string(index=False)}')

def save_preview(tests, pred, out):
    from PIL import Image, ImageDraw
    tile_w, tile_h, cols = (300, 240, 4)
    canvas = Image.new('RGB', (tile_w * cols, tile_h * math.ceil(len(tests) / cols)), '#161616')
    for i, (record, xy) in enumerate(zip(tests, pred)):
        image = Image.fromarray(read_rgb(record['path']))
        w, h = image.size
        image.thumbnail((tile_w, tile_h - 24))
        draw = ImageDraw.Draw(image)
        if np.any(xy != 0):
            x, y = (xy[0] * image.width / w, xy[1] * image.height / h)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline='lime', width=2)
            draw.line((x - 9, y, x + 9, y), fill='lime', width=1)
            draw.line((x, y - 9, x, y + 9), fill='lime', width=1)
        ox, oy = (i % cols * tile_w, i // cols * tile_h)
        canvas.paste(image, (ox, oy))
        ImageDraw.Draw(canvas).text((ox + 5, oy + tile_h - 20), f"{record['key']}: X={xy[0]:.1f} Y={xy[1]:.1f}", fill='white')
    canvas.save(out / 'test_predictions.jpg', quality=90)

def raw_image(record):
    return record.get('raw') if record.get('raw') is not None else read_rgb(record['path'])

def cache_raw_images(records, enabled):
    if enabled:
        for record in records:
            if record.get('raw') is None:
                record['raw'] = read_rgb(record['path'])

def crop_affine(center, width, size, angle=0.0):
    center = np.asarray(center, dtype=np.float64)
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(angle), size / width)
    matrix[:, 2] += (size - 1) / 2 - center
    return matrix

class RefineDataset(Dataset):

    def __init__(self, records, args, centers=None, train=False, scale=1.0):
        self.records, self.args, self.centers = (records, args, centers)
        self.train, self.scale = (train, scale)

    def __len__(self):
        return len(self.records) * (self.args.refine_repeat if self.train else 1)

    def __getitem__(self, index):
        index %= len(self.records)
        record = self.records[index]
        image = raw_image(record)
        size = self.args.refine_size
        point = np.asarray(record['xy'], dtype=np.float32)
        if self.train:
            width = self.args.roi_size * random.uniform(0.85, 1.15)
            sigma = 0.22 if random.random() < 0.2 else 0.12
            displacement = np.clip(np.random.normal(0, sigma, 2), -0.3, 0.3) * width
            center = point + displacement
            matrix = crop_affine(center, width, size, random.uniform(-12, 12))
        else:
            width = self.args.roi_size * self.scale
            center = np.asarray(self.centers[index], dtype=np.float32)
            matrix = crop_affine(center, width, size)
        patch = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        normalized = matrix @ np.r_[point, 1.0] / (size - 1)
        if self.train and random.random() < 0.5:
            patch = patch[:, ::-1].copy()
            normalized[0] = 1 - normalized[0]
        patch = patch.astype(np.float32) / 255.0
        if self.train:
            patch = np.clip(patch * random.uniform(0.85, 1.15), 0, 1) ** random.uniform(0.85, 1.15)
            gray = patch.mean(axis=2, keepdims=True)
            patch = np.clip(gray + random.uniform(0.85, 1.15) * (patch - gray), 0, 1)
            if random.random() < 0.1:
                patch = np.clip(patch + np.random.normal(0, 0.005, patch.shape).astype(np.float32), 0, 1)
        patch = ((patch - MEAN) / STD).transpose(2, 0, 1)
        inverse = cv2.invertAffineTransform(matrix).astype(np.float32)
        return dict(image=torch.from_numpy(np.ascontiguousarray(patch, dtype=np.float32)), point=torch.tensor(normalized, dtype=torch.float32), visible=torch.tensor(record['visible'], dtype=torch.float32), inverse=torch.tensor(inverse), bounds=torch.tensor([image.shape[1] - 1, image.shape[0] - 1], dtype=torch.float32))

def refine_loader(records, args, centers=None, train=False, scale=1.0, seed=0):
    dataset = RefineDataset(records, args, centers, train, scale)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=args.workers, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_seed, generator=torch.Generator().manual_seed(seed), drop_last=False)

def local_heatmap_points(logits, radius=5):
    logits = logits.float()
    batch, _, height, width = logits.shape
    flat = logits.flatten(1)
    peak = flat.argmax(1)
    py, px = (peak // width, peak % width)
    yy, xx = torch.meshgrid(torch.arange(height, device=logits.device), torch.arange(width, device=logits.device), indexing='ij')
    mask = (abs(xx[None] - px[:, None, None]) <= radius) & (abs(yy[None] - py[:, None, None]) <= radius)
    local = logits[:, 0].masked_fill(~mask, -1000000000.0).flatten(1).softmax(1).reshape(batch, height, width)
    x = (local.sum(1) * torch.linspace(0, 1, width, device=logits.device)).sum(1)
    y = (local.sum(2) * torch.linspace(0, 1, height, device=logits.device)).sum(1)
    return torch.stack([x, y], 1)

@torch.no_grad()
def predict_refine(model, records, centers, args, device, scales=None):
    model.eval()
    scales = ([1.0] if args.no_tta else [0.85, 1.0, 1.15]) if scales is None else scales
    all_global, all_local = ([], [])
    for scale in scales:
        predictions_global, predictions_local = ([], [])
        loader = refine_loader(records, args, centers=centers, scale=scale)
        for batch in loader:
            image = batch['image'].to(device, non_blocking=True)
            batch_global, batch_local = ([], [])
            for flip in [False] if args.no_tta else [False, True]:
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    heat, _ = model(image.flip(-1) if flip else image)
                global_xy, local_xy = (heatmap_points(heat), local_heatmap_points(heat))
                if flip:
                    global_xy[:, 0] = 1 - global_xy[:, 0]
                    local_xy[:, 0] = 1 - local_xy[:, 0]
                batch_global.append(global_xy.cpu().numpy())
                batch_local.append(local_xy.cpu().numpy())
            inverse = batch['inverse'].numpy()
            bounds = batch['bounds'].numpy()
            for values, collector in [(batch_global, predictions_global), (batch_local, predictions_local)]:
                points = np.mean(values, axis=0) * (args.refine_size - 1)
                original = np.einsum('bij,bj->bi', inverse[:, :, :2], points) + inverse[:, :, 2]
                collector.append(np.maximum(0, np.minimum(original, bounds)))
        all_global.append(np.concatenate(predictions_global))
        all_local.append(np.concatenate(predictions_local))
    return (np.mean(all_global, axis=0), np.mean(all_local, axis=0))

@torch.no_grad()
def update_ema(average, model, decay):
    for target, current in zip(average.parameters(), model.parameters()):
        target.lerp_(current, 1 - decay)
    for target, current in zip(average.buffers(), model.buffers()):
        target.copy_(current)

def train_refine_fold(fold, train_records, val_records, centers, coarse_state, args, device, out, signature):
    seed_all(args.seed + 1000 + fold)
    checkpoint, marker = (out / f'refine_fold_{fold}.pth', out / f'refine_fold_{fold}_complete.json')
    if args.resume and checkpoint.exists() and marker.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['signature'] != signature:
            raise RuntimeError('Refinement checkpoint configuration mismatch; use a new output directory.')
        model = FoveaNet().to(device)
        model.load_state_dict(saved['model'])
        global_xy, local_xy = predict_refine(model, val_records, centers, args, device)
        del model, saved
        return (global_xy, local_xy)
    if marker.exists():
        marker.unlink()
    train_records = [r for r in train_records if r['visible']]
    if not train_records:
        raise ValueError('No visible training samples for refinement.')
    model = FoveaNet().to(device)
    model.load_state_dict(coarse_state)
    average = copy.deepcopy(model).eval().requires_grad_(False)
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    other = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': args.refine_lr * 0.1}, {'params': other, 'lr': args.refine_lr}], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.refine_epochs, eta_min=args.refine_lr * 0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    loader = refine_loader(train_records, args, train=True, seed=args.seed + fold + 1000)
    truth = np.stack([r['xy'] for r in val_records])
    visible = np.array([r['visible'] > 0 for r in val_records])
    if not visible.any():
        raise ValueError('A validation fold has no visible fovea samples.')
    best, bad, updates, history = (float('inf'), 0, 0, [])
    for epoch in range(1, args.refine_epochs + 1):
        started = time.time()
        model.train()
        for module in model.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        loss_sum, seen = (0.0, 0)
        for batch in loader:
            image = batch['image'].to(device, non_blocking=True)
            point = batch['point'].to(device, non_blocking=True)
            vis = batch['visible'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                heat, vis_logits = model(image)
                loss = training_loss(heat, vis_logits, point, vis, False, args.refine_sigma)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Refinement loss is not finite: fold={fold}, epoch={epoch}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            updates += 1
            update_ema(average, model, min(args.refine_ema, 1 - 1 / (updates + 1)))
            loss_sum += float(loss.detach()) * len(image)
            seen += len(image)
        scheduler.step()
        global_xy, local_xy = predict_refine(average, val_records, centers, args, device)
        score = rmse(0.5 * global_xy[visible] + 0.5 * local_xy[visible], truth[visible])
        history.append(dict(epoch=epoch, train_loss=loss_sum / seen, val_rmse=score, global_rmse=rmse(global_xy[visible], truth[visible]), local_rmse=rmse(local_xy[visible], truth[visible])))
        if score < best:
            best, bad = (score, 0)
            torch.save(dict(model={k: v.detach().cpu().clone() for k, v in average.state_dict().items()}, signature=signature, epoch=epoch, val_rmse=score, image_size=args.refine_size), checkpoint)
        else:
            bad += 1
        print(f'Refine {fold + 1}/{args.folds} Epoch {epoch:03d} loss={loss_sum / seen:.4f} val_RMSE={score:.3f}px best={best:.3f}px time={time.time() - started:.1f}s', flush=True)
        if epoch >= args.refine_min_epochs and bad >= args.refine_patience:
            break
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    average.load_state_dict(saved['model'])
    global_xy, local_xy = predict_refine(average, val_records, centers, args, device)
    pd.DataFrame(history).to_csv(out / f'refine_fold_{fold}_history.csv', index=False)
    write_json(marker, dict(signature=signature, best_epoch=saved['epoch'], val_rmse=best))
    del model, average, saved, optimizer, scheduler, scaler, loader
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return (global_xy, local_xy)

def validate_reused_coarse(folder, records, folds, args):
    folder = Path(folder)
    config_path, oof_path = (folder / 'config.json', folder / 'oof_predictions.csv')
    if not config_path.exists() or not oof_path.exists():
        raise ValueError('Reuse requires config.json and oof_predictions.csv with the five fold checkpoints.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    for field in ['image_size', 'folds', 'seed']:
        if config.get(field) != getattr(args, field):
            raise ValueError(f'Reused coarse checkpoints have incompatible {field}.')
    if config.get('smoke'):
        raise ValueError('Smoke checkpoints cannot be reused for a scored submission.')
    if bool(config.get('fundus_crop', False)) != FUNDUS_CROP:
        raise ValueError('Coarse checkpoint preprocessing differs. Original v1 checkpoints require --no-fundus-crop.')
    old = pd.read_csv(oof_path, dtype={'data': str})
    rows = {normalize_id(row['data']): row for _, row in old.iterrows()}
    if len(rows) != len(records):
        raise ValueError('The reused checkpoints have different training records.')
    for fold, ids in enumerate(folds):
        if not (folder / f'fold_{fold}.pth').exists():
            raise FileNotFoundError(folder / f'fold_{fold}.pth')
        for i in ids:
            r = records[i]
            old_row = rows.get(r['key'])
            if old_row is None or int(old_row['fold']) != fold:
                raise ValueError('The reused checkpoints have different fold assignments.')
            if not np.allclose([old_row['true_x'], old_row['true_y']], r['xy'], atol=0.001, rtol=0):
                raise ValueError('The reused checkpoints have different training coordinates.')
    return folder

def choose_blend(coarse, refined, truth):
    candidates = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    scores = np.array([rmse((1 - weight) * coarse + weight * refined, truth) for weight in candidates])
    best = np.flatnonzero(scores <= scores.min() + 0.02)[0]
    return (float(candidates[best]), scores.tolist())

def run_branch(args=None):
    global FUNDUS_CROP
    args = parse_args() if args is None else args
    FUNDUS_CROP = not args.no_fundus_crop
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(0)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f'Device={device}; coarse={args.image_size}; refine={args.refine_size}; ROI={args.roi_size}', flush=True)
    if args.smoke:
        print('SMOKE MODE: flow validation only; do not submit its predictions.', flush=True)
    elif device.type == 'cpu':
        warnings.warn('GPU is unavailable. Enable GPU in Kaggle for full training.')
    root = find_data(args, out)
    records, tests, sample = read_records(root, args.image_size)
    negative = sum((r['visible'] == 0 for r in records))
    positive = len(records) - negative
    if not positive:
        raise ValueError('No visible training samples.')
    use_visibility = negative > 0
    if use_visibility and min(positive, negative) < args.folds:
        raise ValueError('Reduce folds so each visibility class has at least that many samples.')
    if not use_visibility:
        print('No invisible labels in training. Visibility classification is disabled.', flush=True)
    folds = make_folds(records, args.folds, args.split_seed)
    coarse_source = validate_reused_coarse(args.coarse_checkpoints, records, folds, args) if args.coarse_checkpoints else out
    encoder_state, init_description = (None, 'reused trained coarse checkpoints') if args.coarse_checkpoints else get_encoder_state(args)
    print('Encoder:', init_description, flush=True)
    config = vars(args).copy()
    config.update(data_root=str(root), encoder_init=init_description, use_visibility=use_visibility, n_positive=positive, n_negative=negative, torch_version=str(torch.__version__), fundus_crop=FUNDUS_CROP, backbone='resnet34', method='multi_resolution_v4_branch')
    model_config = {k: v for k, v in config.items() if k not in ['output_dir', 'resume', 'workers', 'data_root', 'experiment_fold', 'model_source', 'baseline_only', 'alternative_size', 'alternative_refine_size', 'alternative_seed']}
    signature = hashlib.sha256((dataset_signature(records, tests) + json.dumps(model_config, sort_keys=True)).encode()).hexdigest()
    config['signature'] = signature
    write_json(out / 'config.json', config)
    cache_raw_images(records, not args.no_image_cache)
    n = len(records)
    coarse_oof, fine_global_oof, fine_local_oof = (np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2)))
    probabilities, fold_ids, used = (np.zeros(n), np.full(n, -1, dtype=int), np.zeros(n, dtype=bool))
    selected = range(args.folds) if args.experiment_fold < 0 else [args.experiment_fold]
    for fold in selected:
        ids = folds[fold]
        train_ids = np.setdiff1d(np.arange(n), ids)
        training, validation = ([records[i] for i in train_ids], [records[i] for i in ids])
        if not len(training) or not len(validation):
            raise ValueError('An empty fold was created.')
        if args.coarse_checkpoints:
            saved = torch.load(coarse_source / f'fold_{fold}.pth', map_location='cpu', weights_only=True)
            model = FoveaNet().to(device)
            model.load_state_dict(saved['model'])
            coarse_xy, probability = predict(model, make_loader(validation, args), device, args, use_visibility)
            coarse_state = saved['model']
            del model, saved
        else:
            coarse_xy, probability = train_fold(fold, training, validation, encoder_state, args, device, use_visibility, out, signature)
            coarse_state = torch.load(out / f'fold_{fold}.pth', map_location='cpu', weights_only=True)['model']
        fine_global, fine_local = train_refine_fold(fold, training, validation, coarse_xy, coarse_state, args, device, out, signature)
        coarse_oof[ids], fine_global_oof[ids], fine_local_oof[ids] = (coarse_xy, fine_global, fine_local)
        probabilities[ids], fold_ids[ids], used[ids] = (probability, fold, True)
        del coarse_state
        gc.collect()
    truth = np.stack([r['xy'] for r in records])
    coarse = apply_visibility(coarse_oof[used], probabilities[used], 0.5, use_visibility)
    fine = apply_visibility(0.5 * fine_global_oof[used] + 0.5 * fine_local_oof[used], probabilities[used], 0.5, use_visibility)
    weight, grid_scores = choose_blend(coarse, fine, truth[used])
    blended = (1 - weight) * coarse + weight * fine
    crossfit = np.zeros_like(coarse)
    crossfit_weights = {}
    observed_folds = fold_ids[used]
    if args.experiment_fold < 0:
        for fold in range(args.folds):
            fit, evaluate = (observed_folds != fold, observed_folds == fold)
            w, _ = choose_blend(coarse[fit], fine[fit], truth[used][fit])
            crossfit[evaluate] = (1 - w) * coarse[evaluate] + w * fine[evaluate]
            crossfit_weights[str(fold)] = w
    else:
        crossfit[:] = blended
    candidate_weight = weight
    candidate_score = rmse(blended, truth[used])
    crossfit_score = rmse(crossfit, truth[used]) if args.experiment_fold < 0 else None
    if crossfit_score is not None and crossfit_score >= rmse(coarse, truth[used]):
        weight = 0.0
        blended = coarse.copy()
    summary = dict(coarse_oof_rmse=rmse(coarse, truth[used]), refine_oof_rmse=rmse(fine, truth[used]), selected_blend_oof_rmse=rmse(blended, truth[used]), crossfit_blend_oof_rmse=crossfit_score, candidate_refine_weight=candidate_weight, candidate_blend_oof_rmse=candidate_score, refine_weight=weight, blend_grid=[0.0, 0.25, 0.5, 0.75, 1.0], blend_grid_rmse=grid_scores, crossfit_weights=crossfit_weights, smoke=args.smoke, experiment_fold=args.experiment_fold, note='Internal validation only; not a leaderboard score. Epoch selection uses each validation fold.')
    write_json(out / 'validation_summary.json', summary)
    print('\nVALIDATION:', json.dumps(summary, ensure_ascii=False), flush=True)
    pd.DataFrame(dict(data=[r['key'] for r, ok in zip(records, used) if ok], fold=observed_folds, true_x=truth[used, 0], true_y=truth[used, 1], coarse_x=coarse[:, 0], coarse_y=coarse[:, 1], refine_x=fine[:, 0], refine_y=fine[:, 1], pred_x=blended[:, 0], pred_y=blended[:, 1], crossfit_x=crossfit[:, 0], crossfit_y=crossfit[:, 1])).to_csv(out / 'oof_predictions.csv', index=False)
    if args.experiment_fold >= 0:
        print('Single-fold experiment complete. No submission was generated.', flush=True)
        return summary
    cache_raw_images(tests, not args.no_image_cache)
    test_loader = make_loader(tests, args)
    fold_coarse, fold_fine, fold_probability = ([], [], [])
    for fold in range(args.folds):
        saved = torch.load(coarse_source / f'fold_{fold}.pth', map_location='cpu', weights_only=True)
        model = FoveaNet().to(device)
        model.load_state_dict(saved['model'])
        coarse_xy, probability = predict(model, test_loader, device, args, use_visibility)
        del saved
        saved = torch.load(out / f'refine_fold_{fold}.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model'])
        fine_global, fine_local = predict_refine(model, tests, coarse_xy, args, device)
        fold_coarse.append(coarse_xy)
        fold_fine.append(0.5 * fine_global + 0.5 * fine_local)
        fold_probability.append(probability)
        del saved, model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    probability = np.mean(fold_probability, axis=0)
    test_coarse = apply_visibility(np.mean(fold_coarse, axis=0), probability, 0.5, use_visibility)
    test_fine = apply_visibility(np.mean(fold_fine, axis=0), probability, 0.5, use_visibility)
    final = (1 - weight) * test_coarse + weight * test_fine
    save_submission(sample, tests, final, out)
    diagnostic = pd.DataFrame(dict(data=[r['key'] for r in tests], coarse_x=test_coarse[:, 0], coarse_y=test_coarse[:, 1], refine_x=test_fine[:, 0], refine_y=test_fine[:, 1], Fovea_X=final[:, 0], Fovea_Y=final[:, 1]))
    diagnostic.to_csv(out / 'test_coordinates.csv', index=False, float_format='%.9f')
    save_preview(tests, final, out)
    print(f'Done. Outputs: {out.resolve()}', flush=True)
    return summary

def check_model_source(folder, args, records):
    folder = Path(folder).resolve()
    required = ['config.json', 'validation_summary.json', 'oof_predictions.csv']
    required += [f'fold_{fold}.pth' for fold in range(args.folds)]
    required += [f'refine_fold_{fold}.pth' for fold in range(args.folds)]
    if not all((folder / name).is_file() for name in required):
        raise ValueError('Model source must include config, validation summary, OOF and all coarse/refine checkpoints.')
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if config.get('backbone', 'resnet34') != 'resnet34':
        raise ValueError('The baseline source must be the previous ResNet34 model.')
    if bool(config.get('smoke', False)) != bool(args.smoke):
        raise ValueError('The model source smoke/formal setting differs from this run.')
    if bool(config.get('fundus_crop', False)) != (not args.no_fundus_crop):
        raise ValueError('The model source foreground preprocessing differs.')
    for key in ['image_size', 'refine_size', 'roi_size', 'folds', 'seed', 'no_tta']:
        if config.get(key) != getattr(args, key):
            raise ValueError(f'Model source configuration mismatch: {key}')
    if config.get('split_seed', config.get('seed')) != args.split_seed:
        raise ValueError('The source has a different validation split.')
    oof = pd.read_csv(folder / 'oof_predictions.csv', dtype={'data': str})
    if len(oof) != len(records) or not {'data', 'fold', 'true_x', 'true_y'}.issubset(oof.columns):
        raise ValueError('The source OOF records are incomplete.')
    rows = {normalize_id(row['data']): row for _, row in oof.iterrows()}
    if len(rows) != len(records):
        raise ValueError('The source contains duplicated training IDs.')
    for fold, ids in enumerate(make_folds(records, args.folds, args.split_seed)):
        for i in ids:
            r, row = records[i], rows.get(records[i]['key'])
            if row is None or int(row['fold']) != fold:
                raise ValueError('The source has a different fold assignment.')
            if not np.allclose([row['true_x'], row['true_y']], r['xy'], atol=.001, rtol=0):
                raise ValueError('The source has different coordinate annotations.')
        for prefix in ['fold_', 'refine_fold_']:
            marker = folder / f'{prefix}{fold}_complete.json'
            if marker.exists():
                data = json.loads(marker.read_text(encoding='utf-8'))
                if data.get('signature') != config.get('signature'):
                    raise ValueError('The source mixes incomplete or incompatible training runs.')
    summary = json.loads((folder / 'validation_summary.json').read_text(encoding='utf-8'))
    weight = summary.get('refine_weight')
    if not isinstance(weight, (float, int)) or not 0 <= weight <= 1:
        raise ValueError('The source refinement weight is invalid.')
    return folder


def resolve_model_source(args, records):
    if args.model_source not in ['', 'auto']:
        return check_model_source(args.model_source, args, records)
    if args.model_source == '':
        return None
    candidates = [Path('/kaggle/working/task3_optimized'), Path(args.output_dir) / f'base_{args.image_size}']
    if Path('/kaggle/input').exists():
        candidates += [p.parent for p in sorted(Path('/kaggle/input').rglob('config.json'))]
    checked = set()
    for folder in candidates:
        resolved = str(folder.resolve())
        if resolved in checked or not folder.exists():
            continue
        checked.add(resolved)
        try:
            return check_model_source(folder, args, records)
        except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
            continue
    return None


def reuse_baseline(source, args, out):
    global FUNDUS_CROP
    FUNDUS_CROP = not args.no_fundus_crop
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = find_data(args, out)
    records, tests, sample = read_records(root, args.image_size)
    source = check_model_source(source, args, records)
    source_config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    summary = json.loads((source / 'validation_summary.json').read_text(encoding='utf-8'))
    weight = float(summary['refine_weight'])
    use_visibility = any(r['visible'] == 0 for r in records)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_raw_images(records + tests, not args.no_image_cache)
    prediction = np.zeros((len(records), 2))
    fold_ids = np.full(len(records), -1, dtype=int)
    test_predictions = []
    for fold, ids in enumerate(make_folds(records, args.folds, args.split_seed)):
        model = FoveaNet().to(device)
        coarse = torch.load(source / f'fold_{fold}.pth', map_location='cpu', weights_only=True)
        if coarse.get('signature') != source_config.get('signature'):
            raise ValueError('Coarse source checkpoint signature mismatch.')
        model.load_state_dict(coarse['model'])
        del coarse
        validation = [records[i] for i in ids]
        val_xy, val_probability = predict(model, make_loader(validation, args), device, args, use_visibility)
        test_xy, test_probability = predict(model, make_loader(tests, args), device, args, use_visibility)
        if weight > 0:
            refined = torch.load(source / f'refine_fold_{fold}.pth', map_location='cpu', weights_only=True)
            if refined.get('signature') != source_config.get('signature'):
                raise ValueError('Refinement source checkpoint signature mismatch.')
            model.load_state_dict(refined['model'])
            del refined
            vg, vl = predict_refine(model, validation, val_xy, args, device)
            tg, tl = predict_refine(model, tests, test_xy, args, device)
            val_xy = (1 - weight) * val_xy + weight * .5 * (vg + vl)
            test_xy = (1 - weight) * test_xy + weight * .5 * (tg + tl)
        prediction[ids] = apply_visibility(val_xy, val_probability, .5, use_visibility)
        fold_ids[ids] = fold
        test_predictions.append(apply_visibility(test_xy, test_probability, .5, use_visibility))
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'Reused baseline fold {fold + 1}/{args.folds}', flush=True)
    truth = np.stack([r['xy'] for r in records])
    oof = pd.DataFrame(dict(data=[r['key'] for r in records], fold=fold_ids,
                            true_x=truth[:, 0], true_y=truth[:, 1], pred_x=prediction[:, 0], pred_y=prediction[:, 1]))
    test_prediction = np.mean(test_predictions, axis=0)
    coordinates = pd.DataFrame(dict(data=[r['key'] for r in tests], Fovea_X=test_prediction[:, 0], Fovea_Y=test_prediction[:, 1]))
    oof.to_csv(out / 'oof_predictions.csv', index=False)
    coordinates.to_csv(out / 'test_coordinates.csv', index=False, float_format='%.9f')
    summary['reused_from'] = str(source)
    summary['recomputed_oof_rmse'] = rmse(prediction, truth)
    write_json(out / 'validation_summary.json', summary)
    source_config['reused_from'] = str(source)
    write_json(out / 'config.json', source_config)
    save_submission(sample, tests, test_prediction, out)
    return oof, coordinates


def read_branch_output(folder):
    folder = Path(folder)
    return (pd.read_csv(folder / 'oof_predictions.csv', dtype={'data': str}),
            pd.read_csv(folder / 'test_coordinates.csv', dtype={'data': str}) if (folder / 'test_coordinates.csv').exists() else None)


def merge_branch_oof(baseline, alternative):
    baseline, alternative = baseline.copy(), alternative.copy()
    baseline['data'] = baseline['data'].map(normalize_id)
    alternative['data'] = alternative['data'].map(normalize_id)
    if baseline['data'].duplicated().any() or alternative['data'].duplicated().any():
        raise ValueError('Duplicated branch prediction IDs.')
    joined = baseline.merge(alternative, on='data', how='inner', suffixes=('_base', '_zoom'), validate='one_to_one')
    if len(joined) != len(baseline) or len(joined) != len(alternative):
        raise ValueError('Branch OOF image IDs differ.')
    if not np.array_equal(joined['fold_base'], joined['fold_zoom']):
        raise ValueError('Branch validation folds differ.')
    truth = joined[['true_x_base', 'true_y_base']].to_numpy()
    if not np.allclose(truth, joined[['true_x_zoom', 'true_y_zoom']], atol=.001, rtol=0):
        raise ValueError('Branch annotations differ.')
    return joined, truth, joined[['pred_x_base', 'pred_y_base']].to_numpy(), joined[['pred_x_zoom', 'pred_y_zoom']].to_numpy()


def crossvalidated_model_blend(baseline, alternative, truth, folds):
    candidate, grid = choose_blend(baseline, alternative, truth)
    unique = np.unique(folds)
    base_score = rmse(baseline, truth)
    equal_prediction = .5 * baseline + .5 * alternative
    equal_score = rmse(equal_prediction, truth)
    crossfit, selected = np.zeros_like(baseline), {}
    fold_scores = {}
    equal_better, adaptive_better = 0, 0
    for fold in unique:
        evaluate = folds == fold
        b = rmse(baseline[evaluate], truth[evaluate])
        e = rmse(equal_prediction[evaluate], truth[evaluate])
        equal_better += int(e < b)
        fold_scores[str(int(fold))] = dict(baseline_rmse=b, equal_blend_rmse=e)
    if len(unique) > 1:
        for fold in unique:
            fit, evaluate = folds != fold, folds == fold
            w, _ = choose_blend(baseline[fit], alternative[fit], truth[fit])
            crossfit[evaluate] = (1 - w) * baseline[evaluate] + w * alternative[evaluate]
            selected[str(int(fold))] = w
            score = rmse(crossfit[evaluate], truth[evaluate])
            adaptive_better += int(score < fold_scores[str(int(fold))]['baseline_rmse'])
            fold_scores[str(int(fold))]['adaptive_crossfit_rmse'] = score
        cv_score = rmse(crossfit, truth)
    else:
        cv_score = None
    weight, mode, selected_score = 0., 'baseline', base_score
    required = len(unique) // 2 + 1
    minimum_gain = max(.05, .005 * base_score)
    if equal_score < base_score - minimum_gain and equal_better >= required:
        weight, mode, selected_score = .5, 'fixed_equal', equal_score
    if cv_score is not None and candidate > 0 and cv_score < selected_score - minimum_gain and adaptive_better >= required:
        weight, mode = candidate, 'adaptive_crossfit'
    if len(unique) == 1 and rmse((1 - candidate) * baseline + candidate * alternative, truth) < selected_score - minimum_gain:
        weight, mode = candidate, 'single_fold_experiment'
    final = (1 - weight) * baseline + weight * alternative
    summary = dict(baseline_oof_rmse=base_score, zoom_oof_rmse=rmse(alternative, truth),
                   equal_blend_oof_rmse=equal_score, candidate_zoom_weight=candidate, zoom_weight=weight,
                   selection_mode=mode, minimum_gain=minimum_gain, equal_blend_improved_folds=equal_better,
                   adaptive_improved_folds=adaptive_better, required_improved_folds=required,
                   fold_scores=fold_scores, final_oof_rmse=rmse(final, truth), crossfit_blend_oof_rmse=cv_score,
                   crossfit_weights=selected, blend_grid=[0., .25, .5, .75, 1.], blend_grid_rmse=grid,
                   note='Fixed equal blending does not fit a weight. Adaptive weights are checked on other folds. Epochs and inner refinement weights use validation data.')
    return final, summary


def main(args=None):
    global FUNDUS_CROP, EXPORT_SUBMISSION
    args = parse_args() if args is None else args
    EXPORT_SUBMISSION = False
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(0)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    seed_all(args.seed)
    FUNDUS_CROP = not args.no_fundus_crop
    root = find_data(args, out)
    records, tests, sample = read_records(root, args.image_size)
    source = resolve_model_source(args, records) if args.experiment_fold < 0 else None
    del records, tests
    baseline_args = copy.deepcopy(args)
    baseline_args.output_dir = str(out / f'base_{args.image_size}')
    if source is not None:
        print(f'Reusing trained ResNet34 baseline: {source}', flush=True)
        baseline_oof, baseline_test = reuse_baseline(source, baseline_args, baseline_args.output_dir)
    else:
        print(f'Training baseline branch at {args.image_size} pixels.', flush=True)
        run_branch(baseline_args)
        baseline_oof, baseline_test = read_branch_output(baseline_args.output_dir)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.baseline_only:
        if baseline_test is not None:
            lookup = {normalize_id(row['data']): row for _, row in baseline_test.iterrows()}
            predictions, ordered_records = [], []
            for item in sample.ImageID:
                key = normalize_id(item.rsplit('_Fovea_', 1)[0])
                if key not in [r['key'] for r in ordered_records]:
                    ordered_records.append(dict(key=key))
                    predictions.append([lookup[key]['Fovea_X'], lookup[key]['Fovea_Y']])
            EXPORT_SUBMISSION = True
            save_submission(sample, ordered_records, np.asarray(predictions), out)
        return
    alternative_args = copy.deepcopy(args)
    alternative_args.seed = args.alternative_seed
    alternative_args.image_size = args.alternative_size
    alternative_args.refine_size = args.alternative_refine_size
    alternative_args.coarse_checkpoints = ''
    alternative_args.output_dir = str(out / f'zoom_{args.alternative_size}')
    print('Training higher-resolution ResNet34 branch with an independent training seed.', flush=True)
    run_branch(alternative_args)
    alternative_oof, alternative_test = read_branch_output(alternative_args.output_dir)
    joined, truth, p34, p50 = merge_branch_oof(baseline_oof, alternative_oof)
    prediction, summary = crossvalidated_model_blend(p34, p50, truth, joined['fold_base'].to_numpy())
    summary.update(smoke=args.smoke, baseline_source=str(source) if source is not None else 'trained in this run',
                   experiment_fold=args.experiment_fold)
    joined['pred_x'], joined['pred_y'] = prediction[:, 0], prediction[:, 1]
    joined.to_csv(out / 'oof_predictions.csv', index=False)
    write_json(out / 'validation_summary.json', summary)
    print('\nFINAL VALIDATION:', json.dumps(summary, ensure_ascii=False), flush=True)
    if args.experiment_fold >= 0:
        print('Single-fold validation only. No submission generated.', flush=True)
        return summary
    if baseline_test is None or alternative_test is None:
        raise RuntimeError('A branch did not generate test coordinates.')
    baseline_test, alternative_test = baseline_test.copy(), alternative_test.copy()
    baseline_test['data'] = baseline_test['data'].map(normalize_id)
    alternative_test['data'] = alternative_test['data'].map(normalize_id)
    test = baseline_test.merge(alternative_test, on='data', suffixes=('_base', '_zoom'), how='inner', validate='one_to_one')
    if len(test) != len(baseline_test) or len(test) != len(alternative_test):
        raise ValueError('Branch test IDs differ.')
    weight = summary['zoom_weight']
    xy = (1 - weight) * test[['Fovea_X_base', 'Fovea_Y_base']].to_numpy() + weight * test[['Fovea_X_zoom', 'Fovea_Y_zoom']].to_numpy()
    ordered_records = [dict(key=str(key)) for key in test['data']]
    EXPORT_SUBMISSION = True
    save_submission(sample, ordered_records, xy, out)
    pd.DataFrame(dict(data=test['data'], Fovea_X=xy[:, 0], Fovea_Y=xy[:, 1])).to_csv(out / 'test_coordinates.csv', index=False, float_format='%.9f')
    write_json(out / 'run_manifest.json', dict(version='v4', branches=['base', 'zoom'], resolutions=[args.image_size, args.alternative_size],
                                              split_seed=args.split_seed, training_seeds=[args.seed, args.alternative_seed],
                                              baseline_source=summary['baseline_source'], submission='/kaggle/working/submission.csv'))
    print(f'Done. Outputs: {out.resolve()}', flush=True)
    return summary


if __name__ == '__main__':
    main()