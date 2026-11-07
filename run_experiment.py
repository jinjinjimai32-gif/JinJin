"""Reproduce every metric and comparison figure without network access."""
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
from src.filters import fixed_median, switching_median, adaptive_median
from src.metrics import psnr, ssim

ROOT = Path(__file__).resolve().parent
DENSITIES = [0.05, 0.15, 0.30]
SEEDS = [2026, 2027, 2028]
METHODS = {'noisy': lambda x:x, 'median3':lambda x:fixed_median(x,3),
           'median5':lambda x:fixed_median(x,5), 'switching3':switching_median,
           'adaptive7':adaptive_median}


def add_noise(image, density, seed):
    rng = np.random.default_rng(seed)
    draw = rng.random(image.shape)
    out = image.copy()
    out[draw < density/2] = 0
    out[(draw >= density/2) & (draw < density)] = 255
    return out, float(np.mean(draw < density)), float(np.mean(out != image))


def font(size):
    for path in ['C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).exists():
            return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def panels(images, labels, path, crop=None):
    cell, top = 320, 48
    canvas = Image.new('RGB', (cell*3, (cell+top)*2), 'white')
    draw = ImageDraw.Draw(canvas)
    for i,(a,label) in enumerate(zip(images,labels)):
        im = Image.fromarray(a)
        if crop: im = im.crop(crop)
        im.thumbnail((cell-12,cell-12), Image.Resampling.NEAREST)
        if crop: im = im.resize((cell-12,cell-12), Image.Resampling.NEAREST)
        x,y=(i%3)*cell,(i//3)*(cell+top)
        draw.text((x+8,y+10),label,font=font(18),fill='black')
        canvas.paste(im,(x+(cell-im.width)//2,y+top+(cell-im.height)//2))
    canvas.save(path)


def main():
    result_dir = ROOT/'results'
    result_dir.mkdir(exist_ok=True)
    rows, manifest = [], []
    for file in sorted((ROOT/'data').glob('*.png')):
        original = Image.open(file).convert('L')
        # Keep aspect ratio; no upsampling. Integer uint8, LANCZOS downsampling.
        image = original.copy()
        image.thumbnail((256,256), Image.Resampling.LANCZOS)
        clean = np.asarray(image)
        manifest.append({'name':file.stem, 'original_size':list(original.size),
                         'processed_size':list(image.size), 'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        Image.fromarray(clean).save(result_dir/f'{file.stem}_clean.png')
        for density in DENSITIES:
            for seed in SEEDS:
                noisy, sampled, changed = add_noise(clean,density,seed)
                outputs = {}
                for method,fn in METHODS.items():
                    start=time.perf_counter()
                    out=fn(noisy)
                    elapsed=(time.perf_counter()-start)*1000
                    outputs[method]=out
                    rows.append(dict(image=file.stem,density=density,seed=seed,method=method,
                                     sampled_fraction=sampled,changed_fraction=changed,
                                     psnr=psnr(clean,out),ssim=ssim(clean,out),runtime_ms=elapsed))
                if density==0.30 and seed==2026:
                    ims=[clean]+list(outputs.values())
                    labels=['Reference','Noisy 30%','Median 3x3','Median 5x5','Switching 3x3','Adaptive max 7x7']
                    panels(ims,labels,result_dir/f'{file.stem}_comparison.png')
                    h,w=clean.shape
                    box=(w//4,h//4,w//4+64,h//4+64)
                    panels(ims,labels,result_dir/f'{file.stem}_crop.png',crop=box)
                    for name,a in outputs.items(): Image.fromarray(a).save(result_dir/f'{file.stem}_{name}.png')
        print(file.stem,'finished',flush=True)
    with (result_dir/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    summary=[]
    for density in DENSITIES:
        for method in METHODS:
            selected=[r for r in rows if r['density']==density and r['method']==method]
            summary.append(dict(density=density,method=method,n=len(selected),
                psnr_mean=float(np.mean([r['psnr'] for r in selected])),
                psnr_std=float(np.std([r['psnr'] for r in selected],ddof=1)),
                ssim_mean=float(np.mean([r['ssim'] for r in selected])),
                runtime_ms_mean=float(np.mean([r['runtime_ms'] for r in selected]))))
    payload=dict(summary=summary,data=manifest,environment=dict(python=platform.python_version(),
        numpy=np.__version__,pillow=PIL.__version__,platform=platform.platform(),processor=platform.processor()),
        protocol=dict(densities=DENSITIES,seeds=SEEDS,max_size=7,padding='symmetric',
                      ssim='7x7 uniform, sample covariance, data_range=255, valid interior'))
    (result_dir/'summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
