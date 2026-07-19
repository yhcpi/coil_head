"""诊断 v17 P2 模型 — 用 model.predict 拿 raw output"""
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

model = YOLO('runs/cfg_truth_repro/v17_hyper_yolo_p2_full/weights/best.pt')

val_imgs = sorted(Path('/home/pi/projects/hyperyolo/data/coil/images/val').glob('*.png'))
samples = [val_imgs[0], val_imgs[10], val_imgs[20]]

print(f'val 图 {len(val_imgs)} 张, sample size: {len(samples)}')

# 让 ultralytics 跑 forward，拿 raw 推理结果
for p in samples:
    print(f'\n=== {p.name} ===')
    # 显式 imgsz=1024 + verbose=False 拿到内部 forward 输出
    result = model.predict(
        source=str(p),
        imgsz=1024,
        conf=0.001,
        max_det=300,
        verbose=False,
        device='cpu',  # 强制 CPU 避免半精度问题
    )
    r = result[0]
    print(f'  shape: {r.orig_shape}, boxes: {len(r.boxes) if r.boxes else 0}')
    if r.boxes is not None and len(r.boxes):
        b = r.boxes
        print(f'  conf range: [{float(b.conf.min()):.4f}, {float(b.conf.max()):.4f}]')
        print(f'  conf mean:  {float(b.conf.mean()):.4f}')
        # xyxy
        xyxy = b.xyxy.cpu().numpy()
        print(f'  boxes xyxy sample (first 3): {xyxy[:3].tolist()}')

print('\n\n=== 改用 forward 拿 raw Detect 输出 ===')
# 直接 forward
import torch
sample = samples[0]
img = Image.open(sample).convert('RGB')
# letterbox 到 1024x1024
from ultralytics.data.augment import LetterBox
lb = LetterBox(new_shape=(1024, 1024), auto=False, scaleup=True, stride=32)
img_arr = np.array(img)
img_lb = lb(image=img_arr)
print(f'letterbox 后 shape: {img_lb.shape}')

x = torch.from_numpy(img_lb.transpose(2, 0, 1)).float().unsqueeze(0).half()
print(f'x shape: {x.shape}, dtype: {x.dtype}')

# raw forward
model.model.eval()
with torch.no_grad():
    # Neck 输出（detect 之前）
    feats = model.model.model[:-1](x)
    print(f'Neck 输出: type={type(feats).__name__}')
    if isinstance(feats, (list, tuple)):
        for i, f in enumerate(feats):
            if isinstance(f, torch.Tensor):
                stride = 1024 // f.shape[2]
                print(f'  scale[{i}] stride={stride} shape={tuple(f.shape)} '
                      f'mean={f.mean():.4f} std={f.std():.4f} max_abs={f.abs().max():.4f}')

    # Detect head
    det = model.model.model[-1]
    raw = det(feats)
    print(f'\nDetect raw output type: {type(raw).__name__}')
    if isinstance(raw, (list, tuple)):
        for i, o in enumerate(raw):
            if isinstance(o, torch.Tensor):
                stride = det.stride[i].item() if i < len(det.stride) else '?'
                # raw shape (B, no, H, W) 通常
                no = o.shape[1] if o.dim() == 4 else o.shape[-1]
                # 取 cls logits (第 5 个起)
                if o.dim() == 4:
                    cls_logits = o[:, 4:, :, :]  # (B, nc, H, W) 或 (B, nc*na, H, W)
                    box_pred = o[:, :4, :, :]
                else:
                    cls_logits = o[..., 4:]
                    box_pred = o[..., :4]
                print(f'  scale[{i}] stride={stride} shape={tuple(o.shape)}')
                print(f'    box_pred: mean={box_pred.mean():.4f} std={box_pred.std():.4f} max_abs={box_pred.abs().max():.4f}')
                print(f'    cls_logits: mean={cls_logits.mean():.4f} std={cls_logits.std():.4f} max={cls_logits.max():.4f} min={cls_logits.min():.4f}')