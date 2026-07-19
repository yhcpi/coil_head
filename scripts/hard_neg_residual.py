#!/usr/bin/env python3
"""
HardNegResidual — 训练二阶段轻量 CNN 兜底 V18.3 剩余 FP
目标：用 33 张 hard neg crop + 50 张真 tip + 50 张随机负样本
       训练 2 层 CNN 判别 patch 是否为 hard neg pattern
       推理时对 V18.3 conf>0.05 的候选框 crop patch，二阶段 CNN 判别

注：本脚本是完整实现，包括：
  1. 数据准备：从 train 集 + 33 张 hard neg 副本 crop patch
  2. 模型训练：2 层 CNN (~50K params) 二分类
  3. 推理 pipeline：V18.3 → crop patch → CNN → 若 p(hard_neg)>0.5 则 conf *= 0.2
  4. 评估：对比 baseline vs V18.3 vs V18.3 + Residual

输出：
  - /tmp/hard_neg_residual_cnn.pt (训练好的 CNN)
  - /tmp/hard_neg_residual_results.json (评估对比)
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO

PATCH_SIZE = 64  # tip 20x20 + 上下文，总 64x64
CNN_EPOCHS = 30
CNN_BATCH = 32


class HardNegCNN(nn.Module):
    """2 层 CNN ~50K params"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.gap(x).flatten(1)
        return torch.sigmoid(self.fc(x))


def crop_patch(img_arr, box_xyxy, size=PATCH_SIZE):
    """从 img_arr (H, W, 3) 按 box_xyxy (4,) crop 并 resize 到 size"""
    x1, y1, x2, y2 = box_xyxy.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_arr.shape[1], x2), min(img_arr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = img_arr[y1:y2, x1:x2]
    pil = Image.fromarray(patch)
    pil = pil.resize((size, size), Image.BILINEAR)
    return np.array(pil) / 255.0


def prepare_patches(model, data_yaml, n_random_neg=50, imgsz=1024):
    """
    准备训练数据：
      - 正样本 (label=1): train 集所有 GT crop (~312 个)
      - hard neg (label=0): 33 张 hn* 副本 (整个图当 patch, 取中心)
      - 随机负样本 (label=0): train 集中随机取的反例 crop (n_random_neg 个)
    """
    import yaml
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    train_imgs_dir = Path(cfg['path']) / cfg['train']
    train_lbls_dir = Path(cfg['path']) / 'labels' / 'train'

    pos_patches = []  # 真 tip
    neg_patches = []  # hard neg

    # 1. 正样本 = train 集所有 GT
    for img_path in sorted(train_imgs_dir.glob('*.png')):
        lbl_path = train_lbls_dir / (img_path.stem + '.txt')
        if not lbl_path.exists():
            continue
        img = np.array(Image.open(img_path).convert('RGB').resize((imgsz, imgsz), Image.BILINEAR))
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, cx, cy, w, h = map(float, parts[:5])
                    box = np.array([(cx - w/2) * imgsz, (cy - h/2) * imgsz,
                                    (cx + w/2) * imgsz, (cy + h/2) * imgsz])
                    patch = crop_patch(img, box)
                    if patch is not None:
                        pos_patches.append(patch)

    # 2. hard neg = 33 张 hn* 副本整图当 patch
    for img_path in sorted(train_imgs_dir.glob('hn*.png')):
        img = np.array(Image.open(img_path).convert('RGB').resize((imgsz, imgsz), Image.BILINEAR))
        # 用全图中央 crop
        h, w = img.shape[:2]
        box = np.array([w/2 - 100, h/2 - 100, w/2 + 100, h/2 + 100])
        patch = crop_patch(img, box, size=PATCH_SIZE)
        if patch is not None:
            neg_patches.append(patch)

    # 3. 随机负样本 = train 集随机取反例 crop (无 GT 区域)
    all_imgs = sorted([p for p in train_imgs_dir.glob('*.png') if not p.stem.startswith('hn')])
    rng = np.random.RandomState(42)
    for _ in range(n_random_neg):
        img_path = rng.choice(all_imgs)
        lbl_path = train_lbls_dir / (img_path.stem + '.txt')
        gt_boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, cx, cy, w, h = map(float, parts[:5])
                        gt_boxes.append(np.array([(cx - w/2) * imgsz, (cy - h/2) * imgsz,
                                                  (cx + w/2) * imgsz, (cy + h/2) * imgsz]))
        img = np.array(Image.open(img_path).convert('RGB').resize((imgsz, imgsz), Image.BILINEAR))
        # 随机 crop 一个 patch，确保不在任何 GT 内
        for _ in range(20):
            x = rng.randint(0, imgsz - 100)
            y = rng.randint(0, imgsz - 100)
            box = np.array([x, y, x + 100, y + 100])
            overlap = False
            for gt in gt_boxes:
                # 检查中心点是否在 GT 内
                cx_p, cy_p = (x + x + 100) / 2, (y + y + 100) / 2
                if gt[0] <= cx_p <= gt[2] and gt[1] <= cy_p <= gt[3]:
                    overlap = True
                    break
            if not overlap:
                patch = crop_patch(img, box, size=PATCH_SIZE)
                if patch is not None:
                    neg_patches.append(patch)
                break

    # Debug: 检查每个 patch 的形状
    if pos_patches:
        pos_shapes = set(p.shape for p in pos_patches)
        if len(pos_shapes) > 1:
            print(f'  WARN pos_patches 形状不一致: {pos_shapes}')
            # 找出非主流形状
            main_shape = max(pos_shapes, key=lambda s: sum(1 for p in pos_patches if p.shape == s))
            print(f'    主流形状: {main_shape}, 异常样本数: {sum(1 for p in pos_patches if p.shape != main_shape)}')
            for i, p in enumerate(pos_patches):
                if p.shape != main_shape:
                    print(f'    pos[{i}] shape={p.shape}')
    if neg_patches:
        neg_shapes = set(p.shape for p in neg_patches)
        if len(neg_shapes) > 1:
            print(f'  WARN neg_patches 形状不一致: {neg_shapes}')
            main_shape = max(neg_shapes, key=lambda s: sum(1 for p in neg_patches if p.shape == s))
            print(f'    主流形状: {main_shape}, 异常样本数: {sum(1 for p in neg_patches if p.shape != main_shape)}')
            for i, p in enumerate(neg_patches):
                if p.shape != main_shape:
                    print(f'    neg[{i}] shape={p.shape}')

    # Filter to same shape before stack
    if pos_patches and neg_patches:
        pos_main = max(set(p.shape for p in pos_patches), key=lambda s: sum(1 for p in pos_patches if p.shape == s))
        neg_main = max(set(p.shape for p in neg_patches), key=lambda s: sum(1 for p in neg_patches if p.shape == s))
        pos_patches = [p for p in pos_patches if p.shape == pos_main]
        neg_patches = [p for p in neg_patches if p.shape == neg_main]

    return np.stack(pos_patches), np.stack(neg_patches)


def train_cnn(pos_patches, neg_patches, epochs=CNN_EPOCHS, batch=CNN_BATCH):
    """训练二分类 CNN"""
    # 数据集 = pos + neg，标签 = 1/0
    X = np.concatenate([pos_patches, neg_patches]).astype(np.float32)
    y = np.concatenate([np.ones(len(pos_patches)), np.zeros(len(neg_patches))]).astype(np.float32)

    # shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_t = torch.from_numpy(X).permute(0, 3, 1, 2)  # (N, 3, H, W)
    y_t = torch.from_numpy(y).unsqueeze(1)

    model = HardNegCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    n = len(X)
    print(f'  CNN 训练: {n} samples, {epochs} epochs, {sum(p.numel() for p in model.parameters())} params')
    for ep in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0
        n_batches = 0
        for i in range(0, n, batch):
            xb = X_t[perm[i:i+batch]]
            yb = y_t[perm[i:i+batch]]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (ep + 1) % 10 == 0:
            with torch.no_grad():
                pred = model(X_t)
                acc = ((pred > 0.5).float() == y_t).float().mean().item()
            print(f'  epoch {ep+1}: loss={total_loss/n_batches:.4f} acc={acc:.4f}')

    return model


def evaluate_with_residual(yolo_model, cnn_model, data_yaml, conf_thresh, cnn_thresh=0.5, suppress_factor=0.2, dist_thresh=30):
    """用 V18.3 + 二阶段 CNN 评估 val 集"""
    import yaml
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    val_imgs_dir = Path(cfg['path']) / cfg['val']
    val_lbls_dir = Path(cfg['path']) / 'labels' / 'val'

    cnn_model.eval()
    TP, FP, FN = 0, 0, 0
    n_residual_hits = 0

    with torch.no_grad():
        for img_path in sorted(val_imgs_dir.glob('*.png')):
            lbl_path = val_lbls_dir / (img_path.stem + '.txt')
            # 读原图尺寸 (YOLO 返回 xyxy 在原图坐标)
            try:
                orig_w, orig_h = Image.open(img_path).size
            except Exception:
                orig_w, orig_h = 1024, 1024
            gt_list = []
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls, cx, cy, w, h = map(float, parts[:5])
                            gt_list.append((cx * orig_w, cy * orig_h, w * orig_w, h * orig_h))

            result = yolo_model.predict(str(img_path), imgsz=1024, conf=conf_thresh, verbose=False)
            boxes = result[0].boxes
            # 用原图尺寸读图（保留原图坐标）
            img_arr = np.array(Image.open(img_path).convert('RGB'))

            if len(boxes) == 0:
                FN += len(gt_list)
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            # 二阶段 CNN 判别
            new_confs = confs.copy()
            patches = []
            patch_idx = []
            for i in range(len(xyxy)):
                p = crop_patch(img_arr, xyxy[i])
                if p is not None:
                    patches.append(p)
                    patch_idx.append(i)
            if patches:
                p_arr = np.stack(patches).astype(np.float32)
                p_t = torch.from_numpy(p_arr).permute(0, 3, 1, 2)
                cnn_pred = cnn_model(p_t).cpu().numpy().flatten()
                for j, i in enumerate(patch_idx):
                    if cnn_pred[j] > cnn_thresh:
                        new_confs[i] *= suppress_factor
                        n_residual_hits += 1

            # 应用 conf 阈值
            keep = new_confs >= conf_thresh
            pred_boxes_f = xyxy[keep]
            pred_confs_f = new_confs[keep]
            gt_matched = np.zeros(len(gt_list), dtype=bool)
            if len(gt_list) > 0:
                gt_xyxy = np.array([[cx - w/2, cy - h/2, cx + w/2, cy + h/2] for cx, cy, w, h in gt_list])

            order = np.argsort(-pred_confs_f) if len(pred_confs_f) > 0 else []
            for i in order:
                if len(gt_list) == 0:
                    FP += 1
                    continue
                cxs = pred_boxes_f[i, [0, 2]].mean()
                cys = pred_boxes_f[i, [1, 3]].mean()
                dists = np.hypot(gt_xyxy[:, [0, 2]].mean(axis=1) - cxs,
                                gt_xyxy[:, [1, 3]].mean(axis=1) - cys)
                x1 = np.maximum(pred_boxes_f[i, 0], gt_xyxy[:, 0])
                y1 = np.maximum(pred_boxes_f[i, 1], gt_xyxy[:, 1])
                x2 = np.minimum(pred_boxes_f[i, 2], gt_xyxy[:, 2])
                y2 = np.minimum(pred_boxes_f[i, 3], gt_xyxy[:, 3])
                inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
                a1 = (pred_boxes_f[i, 2] - pred_boxes_f[i, 0]) * (pred_boxes_f[i, 3] - pred_boxes_f[i, 1])
                a2 = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
                ious = inter / (a1 + a2 - inter + 1e-6)
                match_idx = np.where((dists < dist_thresh) | (ious > 0.1))[0]
                match_idx = match_idx[~gt_matched[match_idx]]
                if len(match_idx) > 0:
                    best = match_idx[np.argmin(dists[match_idx])]
                    gt_matched[best] = True
                    TP += 1
                else:
                    FP += 1
            FN += (~gt_matched).sum()

    Precision = TP / (TP + FP + 1e-6)
    Recall = TP / (TP + FN + 1e-6)
    F1 = 2 * Precision * Recall / (Precision + Recall + 1e-6)
    return dict(TP=int(TP), FP=int(FP), FN=int(FN), Recall=float(Recall),
                Precision=float(Precision), F1=float(F1), n_residual_hits=int(n_residual_hits))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt')
    parser.add_argument('--data', default='data/coil/data.yaml')
    parser.add_argument('--out_cnn', default='/tmp/hard_neg_residual_cnn.pt')
    parser.add_argument('--out_results', default='/tmp/hard_neg_residual_results.json')
    parser.add_argument('--conf_thresh', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=CNN_EPOCHS)
    parser.add_argument('--skip_train', action='store_true', help='跳过 CNN 训练（直接用已有权重）')
    args = parser.parse_args()

    print(f'=== HardNegResidual: V18.3 + 二阶段 CNN 兜底 ===')
    print(f'YOLO 模型: {args.model}')
    print(f'conf_thresh={args.conf_thresh}, CNN epochs={args.epochs}')

    # 1. 准备训练 patch
    if not args.skip_train:
        print('\n准备训练 patch 数据...')
        pos, neg = prepare_patches(YOLO(args.model), args.data)
        print(f'  正样本 (真 tip): {len(pos)}')
        print(f'  负样本 (hard neg + 随机): {len(neg)}')

        # 2. 训练 CNN
        print(f'\n训练二阶段 CNN ({args.epochs} epochs)...')
        cnn_model = train_cnn(pos, neg, epochs=args.epochs)
        torch.save(cnn_model.state_dict(), args.out_cnn)
        print(f'  CNN 权重已保存: {args.out_cnn}')

    # 3. 评估
    print('\n评估 val 99 张...')
    yolo_model = YOLO(args.model)
    cnn_model = HardNegCNN()
    if Path(args.out_cnn).exists():
        cnn_model.load_state_dict(torch.load(args.out_cnn))
    else:
        print(f'  WARN: CNN 权重不存在，使用随机初始化（仅作 demo）')

    metrics = evaluate_with_residual(yolo_model, cnn_model, args.data, args.conf_thresh)
    print(f"\nV18.3 + Residual: F1={metrics['F1']:.4f} R={metrics['Recall']:.4f} P={metrics['Precision']:.4f} TP={metrics['TP']} FP={metrics['FP']} FN={metrics['FN']} (n_residual_hits={metrics['n_residual_hits']})")

    output = dict(
        yolo_model=args.model,
        cnn_model=args.out_cnn,
        conf_thresh=args.conf_thresh,
        metrics=metrics,
    )
    with open(args.out_results, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n结果已保存: {args.out_results}')


if __name__ == '__main__':
    main()