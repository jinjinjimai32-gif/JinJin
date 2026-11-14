import os
import csv
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image

SIZE = 512
PATCH = 256
BATCH = 8
EPOCHS = 140
FOLDS = 5
LR = 1e-4
PATCHES_PER_EPOCH = 256
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = DEVICE == 'cuda'

try:
    import cv2
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
except Exception:
    cv2 = None
    _clahe = None


def find_root():
    env = os.environ.get('SEG_ROOT')
    if env and os.path.isdir(os.path.join(env, 'train', 'image')):
        return env
    for pat in ('/kaggle/input/*/segmentation', '/kaggle/input/*/*/segmentation'):
        for p in sorted(glob.glob(pat)):
            if os.path.isdir(os.path.join(p, 'train', 'image')):
                return p
    for p in sorted(glob.glob('/kaggle/input/**/train/image', recursive=True)):
        return os.path.dirname(os.path.dirname(p))
    return os.path.join(os.getcwd(), 'segmentation')


ROOT = find_root()
OUT = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.getcwd()
IMG_DIR = os.path.join(OUT, 'image')
os.makedirs(IMG_DIR, exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_rgb(path):
    im = Image.open(path).convert('RGB').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def read_mask(path):
    im = Image.open(path).convert('L').resize((SIZE, SIZE), Image.Resampling.NEAREST)
    return (np.asarray(im, dtype=np.float32) > 127).astype(np.float32)


def read_mask_soft(path):
    im = Image.open(path).convert('L').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return (np.asarray(im, dtype=np.float32) > 0).astype(np.float32)


def enhance(img):
    out = img.copy()
    g = out[..., 1]
    if _clahe is not None:
        g = _clahe.apply(np.clip(g * 255.0, 0, 255).astype(np.uint8)).astype(np.float32) / 255.0
    else:
        lo, hi = np.percentile(g, 2), np.percentile(g, 98)
        g = np.clip((g - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    out[..., 1] = g
    return out.astype(np.float32)


def fov_mask(img):
    g = img[..., 1]
    m = (g > 0.05).astype(np.uint8)
    if cv2 is not None:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = (lab == idx).astype(np.uint8)
        m = cv2.dilate(m, np.ones((11, 11), np.uint8))
    return m.astype(np.float32)


def to_tensor(img):
    img = np.ascontiguousarray(img)
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


def augment(img, msk):
    if random.random() < 0.5:
        img = img[:, ::-1]
        msk = msk[:, ::-1]
    if random.random() < 0.5:
        img = img[::-1, :]
        msk = msk[::-1, :]
    k = random.randint(0, 3)
    if k:
        img = np.rot90(img, k)
        msk = np.rot90(msk, k)
    if cv2 is not None:
        img = np.ascontiguousarray(img)
        msk = np.ascontiguousarray(msk)
        h, w = img.shape[:2]
        ang = random.uniform(-30, 30)
        sc = random.uniform(0.85, 1.15)
        tx = random.uniform(-0.06, 0.06) * w
        ty = random.uniform(-0.06, 0.06) * h
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, sc)
        m[0, 2] += tx
        m[1, 2] += ty
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        msk = cv2.warpAffine(msk, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    if random.random() < 0.7:
        img = img * random.uniform(0.8, 1.25) + random.uniform(-0.08, 0.08)
    if random.random() < 0.5:
        img = np.clip(img, 0, 1) ** random.uniform(0.7, 1.4)
    return np.clip(img, 0, 1).astype(np.float32), msk.astype(np.float32)


class TrainSet(Dataset):
    def __init__(self, ids):
        self.ids = list(ids)
        self.data = {}
        for i in self.ids:
            self.data[i] = (enhance(read_rgb(os.path.join(ROOT, 'train', 'image', i))),
                            read_mask(os.path.join(ROOT, 'train', 'label', i)))

    def __len__(self):
        return PATCHES_PER_EPOCH

    def __getitem__(self, idx):
        i = self.ids[np.random.randint(len(self.ids))]
        img, msk = self.data[i]
        img, msk = augment(img.copy(), msk.copy())
        y = np.random.randint(0, SIZE - PATCH + 1)
        x = np.random.randint(0, SIZE - PATCH + 1)
        img = img[y:y + PATCH, x:x + PATCH]
        msk = msk[y:y + PATCH, x:x + PATCH]
        return to_tensor(img), torch.from_numpy(np.ascontiguousarray(msk)).float().unsqueeze(0)


class ValSet(Dataset):
    def __init__(self, ids):
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        img = enhance(read_rgb(os.path.join(ROOT, 'train', 'image', i)))
        msk = read_mask(os.path.join(ROOT, 'train', 'label', i))
        return to_tensor(img), torch.from_numpy(msk).float().unsqueeze(0), i


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, cin_skip, cin_up, cout):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(cin_up, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
        self.block = ConvBlock(cin_skip + cout, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.reduce(x)
        return self.block(torch.cat([x, skip], 1))


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            enc = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        except Exception:
            enc = resnet34(weights=None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.l1 = enc.layer1
        self.l2 = enc.layer2
        self.l3 = enc.layer3
        self.l4 = enc.layer4
        self.d4 = Decoder(256, 512, 256)
        self.d3 = Decoder(128, 256, 128)
        self.d2 = Decoder(64, 128, 64)
        self.d1 = Decoder(64, 64, 64)
        self.up0 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ConvBlock(32, 32),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        c1 = x
        x = self.pool(x)
        c2 = self.l1(x)
        c3 = self.l2(c2)
        c4 = self.l3(c3)
        c5 = self.l4(c4)
        x = self.d4(c5, c4)
        x = self.d3(x, c3)
        x = self.d2(x, c2)
        x = self.d1(x, c1)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up0(x)
        return self.head(x)


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(2, 3)) + 1.0
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1.0
    return 1.0 - (num / den).mean()


def autocast_ctx():
    try:
        return torch.amp.autocast('cuda', enabled=AMP)
    except Exception:
        return torch.cuda.amp.autocast(enabled=AMP)


def make_scaler():
    try:
        return torch.amp.GradScaler('cuda', enabled=AMP)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=AMP)


@torch.no_grad()
def predict_probs(model, img, tta=True):
    model.eval()
    x = img.unsqueeze(0).to(DEVICE)
    if not tta:
        return torch.sigmoid(model(x)).squeeze().float().cpu().numpy()
    probs = 0.0
    for hf in (False, True):
        for vf in (False, True):
            xx = x
            if hf:
                xx = torch.flip(xx, [3])
            if vf:
                xx = torch.flip(xx, [2])
            p = torch.sigmoid(model(xx))
            if vf:
                p = torch.flip(p, [2])
            if hf:
                p = torch.flip(p, [3])
            probs = probs + p
    return (probs / 4.0).squeeze().float().cpu().numpy()


def dice_np(prob, gt, thr):
    pred = (prob > thr).astype(np.float32)
    num = 2.0 * (pred * gt).sum()
    den = pred.sum() + gt.sum()
    if den == 0:
        return 1.0
    return float(num / den)


def train_fold(tr_ids, va_ids, fold):
    model = UNet().to(DEVICE)
    workers = 0 if os.name == 'nt' else 2
    loader = DataLoader(TrainSet(tr_ids), batch_size=BATCH, shuffle=True,
                        num_workers=workers, pin_memory=(DEVICE == 'cuda'), drop_last=True)
    val_loader = DataLoader(ValSet(va_ids), batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=(DEVICE == 'cuda'))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.02)
    scaler = make_scaler()
    bce = nn.BCEWithLogitsLoss()
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for img, msk in loader:
            img = img.to(DEVICE, non_blocking=True)
            msk = msk.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = model(img)
                loss = 0.5 * bce(logits, msk) + 0.5 * dice_loss(logits, msk)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
        sched.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            print('fold %d epoch %d/%d loss %.4f' % (fold, ep + 1, EPOCHS, tot / max(1, len(loader))))
    oof = {}
    scores = []
    for img, msk, name in val_loader:
        p = predict_probs(model, img[0])
        gt = msk[0, 0].numpy()
        oof[name[0]] = p
        scores.append(dice_np(p, gt, 0.5))
    print('fold %d dice %.4f' % (fold, float(np.mean(scores))))
    return model, oof


def rle_encode(binary):
    dots = np.where(binary.reshape(-1) > 0)[0]
    runs = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            runs.extend((int(b) + 1, 0))
        runs[-1] += 1
        prev = b
    return ' '.join(str(v) for v in runs)


def write_submission(path, ids, probs, thr, save_dir=None):
    rows = []
    for k in ids:
        mask = (probs[k] > thr).astype(np.uint8) * 255
        if save_dir is not None:
            Image.fromarray(mask).save(os.path.join(save_dir, k + '.png'))
        rows.append((k, rle_encode(mask > 0)))
    with open(path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Id', 'Predicted'])
        writer.writerows(rows)


def main():
    tr_dir = os.path.join(ROOT, 'train', 'image')
    ids = sorted([f for f in os.listdir(tr_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
                 key=lambda s: int(''.join(c for c in os.path.splitext(s)[0] if c.isdigit()) or 0))
    print('root', ROOT, 'train', len(ids), 'device', DEVICE)
    order = list(range(len(ids)))
    random.Random(SEED).shuffle(order)
    folds = [order[i::FOLDS] for i in range(FOLDS)]
    models = []
    oof = {}
    for f in range(FOLDS):
        va = [ids[i] for i in folds[f]]
        tr = [ids[i] for i in order if i not in set(folds[f])]
        model, pred = train_fold(tr, va, f)
        models.append(model)
        oof.update(pred)

    gts_t = {}
    gts_s = {}
    for i in oof:
        oof[i] = oof[i] * fov_mask(read_rgb(os.path.join(ROOT, 'train', 'image', i)))
        gts_t[i] = read_mask(os.path.join(ROOT, 'train', 'label', i))
        gts_s[i] = read_mask_soft(os.path.join(ROOT, 'train', 'label', i))

    grid = np.arange(0.20, 0.76, 0.02)
    d_t = []
    d_s = []
    for thr in grid:
        d_t.append(float(np.mean([dice_np(oof[i], gts_t[i], thr) for i in oof])))
        d_s.append(float(np.mean([dice_np(oof[i], gts_s[i], thr) for i in oof])))
    d_t = np.array(d_t)
    d_s = np.array(d_s)
    thr_thin = float(grid[int(np.argmax(d_t))])
    thr_soft = float(grid[int(np.argmax(d_s))])
    thr_mid = float(grid[int(np.argmax(d_t + d_s))])
    print('oof thin %.4f at %.2f | soft %.4f at %.2f | mid %.2f (%.4f / %.4f)' % (
        d_t.max(), thr_thin, d_s.max(), thr_soft, thr_mid,
        d_t[int(np.argmax(d_t + d_s))], d_s[int(np.argmax(d_t + d_s))]))

    test_dir = os.path.join(ROOT, 'test', 'image')
    test_ids = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
                      key=lambda s: int(''.join(c for c in os.path.splitext(s)[0] if c.isdigit()) or 0))
    tp = {}
    for name in test_ids:
        raw = read_rgb(os.path.join(test_dir, name))
        x = to_tensor(enhance(raw))
        prob = 0.0
        for model in models:
            prob = prob + predict_probs(model, x)
        tp[os.path.splitext(name)[0]] = prob / len(models) * fov_mask(raw)
    test_keys = [os.path.splitext(n)[0] for n in test_ids]

    write_submission(os.path.join(OUT, 'submission.csv'), test_keys, tp, thr_mid, IMG_DIR)
    write_submission(os.path.join(OUT, 'submission_soft.csv'), test_keys, tp, thr_soft)
    write_submission(os.path.join(OUT, 'submission_thin.csv'), test_keys, tp, thr_thin)
    print('saved submission.csv (thr %.2f), submission_soft.csv (thr %.2f), submission_thin.csv (thr %.2f)'
          % (thr_mid, thr_soft, thr_thin))


if __name__ == '__main__':
    main()