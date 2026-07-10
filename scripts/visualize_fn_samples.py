"""用 best.pt 在 FN 案例上做预测并可视化（GT vs Pred）。"""
import argparse
import os
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, '/home/pi/projects/hyperyolo/repos/Hyper-YOLO')
from ultralytics import YOLO


def load_gt(path):
    if not Path(path).exists():
        return []
    out = []
    for line in open(path).read().strip().split('\n'):
        if not line.strip():
            continue
        cls, cx, cy, w, h = line.split()
        out.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return out


def draw_box(img, xyxy, color, label):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    if label:
        cv2.putText(img, label, (x1, max(y1-8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--img_dir', required=True)
    p.add_argument('--gt_dir', required=True)
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--samples', nargs='+', required=True, help='要可视化的文件名列表')
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    for name in args.samples:
        img_path = Path(args.img_dir) / name
        gt_path = Path(args.gt_dir) / (Path(name).stem + '.txt')

        # 加载原图（BGR）
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]

        # 绘制 GT
        for cls, cx, cy, w, h in load_gt(gt_path):
            x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
            draw_box(img, [x1, y1, x2, y2], (0, 255, 0), f'GT')

        # 预测
        results = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf,
                                max_det=10, verbose=False)[0]
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            # 按 conf 降序，找前 3 个
            order = np.argsort(-confs)[:5]
            for idx, i in enumerate(order):
                x1, y1, x2, y2 = map(int, boxes[i])
                c = float(confs[i])
                draw_box(img, [x1, y1, x2, y2], (0, 0, 255), f'P{idx+1} {c:.2f}')

        # 输出
        out_path = Path(args.out_dir) / name
        cv2.imwrite(str(out_path), img)
        print(f'✓ 已保存：{out_path}')

        # 同时输出原图到 out_dir
        raw_path = Path(args.out_dir) / f'raw_{name}'
        shutil.copy(str(img_path), str(raw_path))
        print(f'  原图：{raw_path}')

    print(f'\n输出目录：{args.out_dir}')


if __name__ == '__main__':
    main()
