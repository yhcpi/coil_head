#!/usr/bin/env python
"""Hard Negative Mining 可视化：把 A/B/C 三类候选图的 GT + 所有 high-conf 预测画到原图上。

输出:
  runs/cfg_truth_repro/v11_baseline_strong_aug_full/hard_neg_viz/
    ├── A_166.png, A_400.png, ...            (5 张 A 类)
    ├── B_556.png, B_493.png, ...            (9 张 B 类)
    ├── C_top30_*.png                        (30 张 C 类)
    └── index.html                            (汇总，可浏览器打开)
"""
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("/home/pi/projects/hyperyolo")
TRAIN_IMG = ROOT / "data/coil/images/train"
TRAIN_LBL = ROOT / "data/coil/labels/train"
CAND = json.load(open("/tmp/v11_hard_neg_candidates.json"))
CATS = json.load(open("/tmp/v11_hard_neg_categories.json"))
OUT_DIR = ROOT / "runs/cfg_truth_repro/v11_baseline_strong_aug_full/hard_neg_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 按图索引候选
by_img = {}
for c in CAND:
    by_img.setdefault(c["img"], []).append(c)

GT_COLOR = (0, 255, 0)        # 绿
PRED_COLOR = (0, 0, 255)      # 红
PRED_TEXT = (255, 255, 255)   # 白
GT_TEXT = (255, 255, 255)     # 白


def load_gt(img_path):
    lbl = TRAIN_LBL / f"{img_path.stem}.txt"
    if not lbl.exists():
        return []
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    boxes = []
    for line in lbl.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        cls, cx, cy, bw, bh = map(float, parts[:5])
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        boxes.append((x1, y1, x2, y2))
    return boxes


def draw(img_path, gt, preds, out_path, title=""):
    img = cv2.imread(str(img_path))
    # 画 GT（绿色实线）
    for g in gt:
        cv2.rectangle(img, (g[0], g[1]), (g[2], g[3]), GT_COLOR, 3)
        cv2.putText(img, "GT", (g[0], max(g[1]-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GT_TEXT, 2)
    # 画 Pred（红色实线，按 conf 排）
    preds_sorted = sorted(preds, key=lambda x: -x["conf"])
    for p in preds_sorted:
        x1, y1, x2, y2 = int(p["bbox"][0]), int(p["bbox"][1]), int(p["bbox"][2]), int(p["bbox"][3])
        cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, 2)
        label = f"pred {p['conf']:.2f}"
        cv2.putText(img, label, (x1, min(y2+18, img.shape[0]-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, PRED_TEXT, 2)
    # 标题栏
    if title:
        cv2.rectangle(img, (0, 0), (img.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(img, title, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), img)


# A 类（全部 5 张）
print("=== A 类（漏标高可疑，5 张）===")
for img_name in CATS["A_漏标高可疑"]:
    img_path = TRAIN_IMG / img_name
    if not img_path.exists():
        continue
    gt = load_gt(img_path)
    preds = sorted(by_img.get(img_name, []), key=lambda x: -x["conf"])
    out = OUT_DIR / f"A_{img_name}"
    draw(img_path, gt, preds, out, title=f"A_漏标高可疑: {img_name} (GT={len(gt)}, pred={len(preds)})")
    top_conf = preds[0]["conf"] if preds else 0
    print(f"  A {img_name}: GT={len(gt)}, pred={len(preds)}, top_conf={top_conf:.3f}")

# B 类（全部 9 张）
print("\n=== B 类（边界可疑，9 张）===")
for img_name in CATS["B_边界可疑"]:
    img_path = TRAIN_IMG / img_name
    if not img_path.exists():
        continue
    gt = load_gt(img_path)
    preds = sorted(by_img.get(img_name, []), key=lambda x: -x["conf"])
    out = OUT_DIR / f"B_{img_name}"
    draw(img_path, gt, preds, out, title=f"B_边界可疑: {img_name} (GT={len(gt)}, pred={len(preds)})")
    top_conf = preds[0]["conf"] if preds else 0
    print(f"  B {img_name}: GT={len(gt)}, pred={len(preds)}, top_conf={top_conf:.3f}")

# C 类 top-30
print("\n=== C 类 top-30（有 GT 但模型别处误报）===")
c_top = CATS["C_有GT但模型别处误报"][:30]
for img_name in c_top:
    img_path = TRAIN_IMG / img_name
    if not img_path.exists():
        continue
    gt = load_gt(img_path)
    preds = sorted(by_img.get(img_name, []), key=lambda x: -x["conf"])
    out = OUT_DIR / f"C_top{img_name}"
    draw(img_path, gt, preds, out, title=f"C_双重目标嫌疑: {img_name} (GT={len(gt)}, pred={len(preds)})")
    top_conf = preds[0]["conf"] if preds else 0
    print(f"  C {img_name}: GT={len(gt)}, pred={len(preds)}, top_conf={top_conf:.3f}")

# 生成 HTML 索引
html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Hard Neg Mining Viz</title>
<style>
body {{ font-family: sans-serif; padding: 20px; background: #f4f4f4; }}
h1 {{ color: #333; }}
h2 {{ color: #666; border-bottom: 2px solid #ccc; padding-bottom: 8px; margin-top: 30px; }}
.row {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.cell {{ background: white; padding: 8px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.cell img {{ max-width: 320px; max-height: 320px; display: block; }}
.cell a {{ color: #0066cc; text-decoration: none; font-size: 0.9em; }}
.cell .gt {{ color: #0a0; }}
.cell .pred {{ color: #c00; }}
</style></head><body>
<h1>Hard Negative Mining 可视化（v11 best.pt on train 545 张）</h1>
<p><b>配色</b>: 绿框 = GT, 红框 = 模型预测 (旁边标 conf)</p>
<p><b>三类别说明</b>:</p>
<ul>
<li><b>A 漏标高可疑</b>: gt_count=0 + conf>0.30 — 最优先重标
<li><b>B 边界可疑</b>: gt_count=0 + conf 0.10-0.30 — 看图确认
<li><b>C 双重目标嫌疑</b>: gt_count=1 + 模型在别处高 conf 预测 — 看图确认是否真有第二目标
</ul>

<h2>A 类 — 漏标高可疑（{len(CATS['A_漏标高可疑'])} 张）</h2>
<div class="row">
""" + "".join(f'<div class="cell"><a href="A_{n}" target="_blank"><img src="A_{n}"><span>{n}</span></a></div>' for n in CATS['A_漏标高可疑']) + """
</div>

<h2>B 类 — 边界可疑（{} 张）</h2>
<div class="row">
""".format(len(CATS['B_边界可疑'])) + "".join(f'<div class="cell"><a href="B_{n}" target="_blank"><img src="B_{n}"><span>{n}</span></a></div>' for n in CATS['B_边界可疑']) + """
</div>

<h2>C 类 — 双重目标嫌疑 top-30（共 {} 张）</h2>
<div class="row">
""".format(len(CATS['C_有GT但模型别处误报'])) + "".join(f'<div class="cell"><a href="C_top{n}" target="_blank"><img src="C_top{n}"><span>{n}</span></a></div>' for n in c_top) + """
</div>

</body></html>"""

(OUT_DIR / "index.html").write_text(html)
print(f"\n=== 输出 ===")
print(f"  可视化目录: {OUT_DIR}")
print(f"  HTML 索引: {OUT_DIR / 'index.html'}")
print(f"  共生成 {len(CATS['A_漏标高可疑']) + len(CATS['B_边界可疑']) + len(c_top)} 张 PNG")