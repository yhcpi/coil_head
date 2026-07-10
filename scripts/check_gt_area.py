from pathlib import Path
from PIL import Image
import numpy as np

areas = []
ws = []
hs = []

val_dir = Path('data/coil/labels')
train_imgs_dir = Path('data/coil/images/train')
val_imgs_dir = Path('data/coil/images/val')

for sub in ['train', 'val']:
    for txt in (val_dir / sub).glob('*.txt'):
        img_path = (train_imgs_dir if sub == 'train' else val_imgs_dir) / (txt.stem + '.png')
        if not img_path.exists():
            continue
        W, H = Image.open(img_path).size
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts)
            px_w = w * W
            px_h = h * H
            areas.append(px_w * px_h)
            ws.append(px_w)
            hs.append(px_h)

if areas:
    a = np.array(areas)
    w = np.array(ws)
    h = np.array(hs)
    print('N=' + str(len(a)) + ' GT bboxes (train+val)')
    print('area px^2: min={:.0f}  median={:.0f}  mean={:.0f}  max={:.0f}'.format(a.min(), np.median(a), a.mean(), a.max()))
    print('  pct >400: {:.1f}%'.format(100 * (a > 400).sum() / len(a)))
    print('  pct >100: {:.1f}%'.format(100 * (a > 100).sum() / len(a)))
    print('  pct >40:  {:.1f}%'.format(100 * (a > 40).sum() / len(a)))
    print('w px:   min={:.1f}  median={:.1f}  max={:.1f}'.format(w.min(), np.median(w), w.max()))
    print('h px:   min={:.1f}  median={:.1f}  max={:.1f}'.format(h.min(), np.median(h), h.max()))
