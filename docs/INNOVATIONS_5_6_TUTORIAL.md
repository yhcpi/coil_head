# 创新点 6（Lenient-mAP）部署手把手教学

> **场景前提**：钢卷头部检测，每张图最多 1 个目标（要么 0 个要么 1 个）。
> **NMS 角色**：在 max_det=1 部署链路上**完全无关**（top1 之后 NMS 不再选谁进谁出），本场景下归档。
> **本教程专注**：创新点 6 怎么部署 + 怎么找业务最佳工作点。
>
> 配套代码：`scripts/lenient_eval.py`、`scripts/scan_top1_thresholds.py`
> 笔记来源：`lenient_label_innovations.md` §1.6
> 旧版（包含创新点 5）已废弃，参见 git history。

---

## 0. 先看清：你的部署链路是什么

```
┌────────────────────────────────────────────────────────────┐
│  2560×1440 大图 → ultralytics model.predict(imgsz=1024)    │
│   输出 raw boxes: [(conf, x1,y1,x2,y2)] × N (N ≤ max_det)  │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │   max_det=1    │  ← 你的场景：每张图最多 1 目标
        │   按 conf 取首 │     NMS 在这一步之后完全没作用
        └────────┬───────┘
                 │  ← top1 = (conf, x1, y1, x2, y2) 或 None
                 ▼
        ┌────────────────┐
        │  conf ≥ thr?   │  ← 业务"有目标/无目标"判定
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │   Lenient 匹配  │  ← 创新点 6 在这
        │ 中心距离 < D    │     vs 标准 IoU≥0.5
        │   算 TP/TN      │
        └────────┬───────┘
                 │
                 ▼
            Recall / Precision / F1
```

**两个关键事实**：
1. **NMS 在 max_det=1 之后没有用武之地**：所有候选都被压成 1 个，IoU-NMS 和 Soft-NMS-Cov 选出的 top1 完全相同。上一版跑的 4 口径矩阵里 N0=N2、N1=N3 就是铁证。
2. **真正决定业务精度的是两个参数**：
   - `--top1_conf_thresh`：conf 阈值（决定"模型说有无目标"）
   - `--dist_thresh_eval`：Lenient 距离阈值（决定"top1 与 GT 算不算命中"）

下面就讲这两个参数怎么选。

---

## 1. 创新点 6：Lenient-mAP（用"中心距离"代替"IoU"做评估与决策）

### 1.1 改了什么、为什么改

| | 标准 IoU-Match | **Lenient-Match** |
|---|---|---|
| **匹配条件** | 预测框与 GT 框 IoU ≥ 0.5 算命中 (TP) | 预测中心到 GT 中心距离 < D 算命中 (TP) |
| **小目标 + 宽容标注** | 10px 偏移就能让 IoU 从 0.7 跌到 0.3 → FP | 10px 偏移距离远小于 D → 仍 TP |
| **物理含义** | "矩形重合度"（受 bbox 大小影响） | "目标中心定位精度"（与目标大小无关） |
| **场景匹配** | 通用大目标、紧 GT | **小目标 + 宽容 GT 标注** |

### 1.2 代码逐行讲解（`scripts/lenient_eval.py`）

#### 关键函数 1：`eval_top1_deploy`（4 种匹配模式）

```python
def eval_top1_deploy(preds, gts, mode='lenient', iou_thresh=0.5,
                     dist_thresh=50, mac_thresh=0.5, conf_thresh=0.001):
    """Per-image top1 部署口径评估。

    mode:
      - 'iou'      : IoU >= iou_thresh 算命中
      - 'lenient'  : 中心距离 < dist_thresh 算命中（创新点 6 主推）
      - 'mac'      : Min-Area Coverage >= mac_thresh 算命中（用户新加指标）
      - 'combined' : lenient OR mac 任一满足即命中（最宽容）
    """

    # 按图分组
    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p[0], []).append(p)
    gts_by_img = {}
    for g in gts:
        gts_by_img.setdefault(g[0], []).append(g)

    tp = fp = fn = tn = 0
    matched_top1_confs, fp_top1_confs = [], []

    # 遍历所有出现过的图（pred 和 gt 都要覆盖）
    all_img_idx = sorted(set(list(preds_by_img.keys()) + list(gts_by_img.keys())))

    for img_idx in all_img_idx:
        img_preds = preds_by_img.get(img_idx, [])
        img_gts   = gts_by_img.get(img_idx, [])

        # ──────── 第 1 步：选 top1（conf 最高） ────────
        top1 = max(img_preds, key=lambda x: x[1]) if img_preds else None

        # ──────── 第 2 步：处理"无 top1"（被 conf_thresh 过滤掉） ────────
        if top1 is None:
            if img_gts: fn += len(img_gts)   # 有 GT 但模型没认出来
            else:       tn += 1              # 真无 GT，模型也沉默，正确
            continue

        # ──────── 第 3 步：top1 与 GT 比较（4 种匹配模式） ────────
        matched_gt = False
        if mode == 'iou':
            best_score = -1
            for g in img_gts:
                iou = compute_iou(top1[2:6], g[2:6])
                if iou > best_score: best_score = iou
            passed = best_score >= iou_thresh
        elif mode == 'mac':     # 新加：Min-Area Coverage
            best_score = -1
            for g in img_gts:
                m = compute_mac(top1[2:6], g[2:6])
                if m > best_score: best_score = m
            passed = best_score >= mac_thresh
        elif mode == 'combined':  # 新加：Lenient OR MAC（最宽容）
            best_dist, best_mac = 1e9, -1
            for g in img_gts:
                d = compute_center_dist(top1[2:6], g[2:6])
                m = compute_mac(top1[2:6], g[2:6])
                if d < best_dist: best_dist = d
                if m > best_mac: best_mac = m
            passed = (best_dist < dist_thresh) or (best_mac >= mac_thresh)
        else:  # lenient（你部署用的就是这条，默认）
            best_score = 1e9
            for g in img_gts:
                d = compute_center_dist(top1[2:6], g[2:6])
                if d < best_score: best_score = d
            passed = best_score < dist_thresh

        # ──────── 第 4 步：分类计数 ────────
        if img_gts:
            if passed: tp += 1; matched_top1_confs.append(top1[1])
            else:      fn += len(img_gts); fp_top1_confs.append(top1[1])
        else:
            fp += 1; fp_top1_confs.append(top1[1])

    recall    = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'recall': recall, 'precision': precision, 'f1': f1,
            'top1_conf_mean_tp': np.mean(matched_top1_confs) if matched_top1_confs else 0,
            'top1_conf_mean_fp': np.mean(fp_top1_confs) if fp_top1_confs else 0}
```

**核心改动只有 3 行**（相对 IoU 版）：

```diff
- best_score = -1                       # IoU：越大越好
+ best_score = 1e9                      # 距离：越小越好

- if iou > best_score: best_score = iou
+ if d < best_score: best_score = d

- passed = best_score >= iou_thresh     # IoU ≥ 0.5
+ passed = best_score < dist_thresh     # 距离 < D
```

#### 关键函数 2：`collect_preds_top1`

```python
def collect_preds_top1(raw_preds_by_image, nms_kind, dist_thresh_nms, iou_thresh):
    """每张图最多 1 个预测（conf 最高）"""
    out = []
    for img_idx, raw in enumerate(raw_preds_by_image):
        if not raw:
            continue
        boxes  = np.array([(r[1], r[2], r[3], r[4]) for r in raw])
        scores = np.array([r[0] for r in raw])

        # NMS 仍然调用，但 max_output=1 让它压成 1 个候选
        if nms_kind == 'iou':
            kb, ks = iou_nms(boxes, scores, iou_thresh=iou_thresh, max_output=1)
        elif nms_kind == 'soft_coverage':
            kb, ks = soft_nms_coverage(boxes, scores,
                                       dist_thresh_px=dist_thresh_nms, max_output=1)
        else:
            order = np.argsort(-scores)[:1]
            kb, ks = boxes[order], scores[order]

        for b, s in zip(kb, ks):
            out.append((img_idx, s, b[0], b[1], b[2], b[3], raw[0][5]))
    return out
```

**注意**：这里 NMS 被调用了，但因为 `max_output=1`，无论 NMS 是 IoU 版还是 Soft-NMS 版，最后都只剩 1 个框。这就是为什么在你的场景下创新点 5（Soft-NMS-Cov）没有贡献——代码可以保留它，但部署评估结果完全相同。

---

## 2. 两个核心参数的物理含义

### 2.1 `--top1_conf_thresh`（业务"有/无目标"判定）

| conf_thr | 含义 |
|---|---|
| 0.001 | 几乎全收，只过滤 conf≈0 的明显垃圾 |
| **0.05** | **建议起步**——能消掉 95% 的误报 |
| 0.2 | 中等严格，可能丢掉一些真目标 |
| 0.5 | 高严格，只信"高把握" |
| 0.7+ | 太严，会把低 conf 但正确的真目标当漏报 |

**怎么选**：
- 跑扫描，看 `FP` 下降的拐点（你的数据上 conf_thr=0.05 是甜点）
- 拐点之前：FP 多；拐点之后：TP 也开始掉

### 2.2 `--dist_thresh_eval`（Lenient 距离阈值）

| dist_thr | 含义 |
|---|---|
| 0 | 等价 IoU≈1，几乎重合 |
| 20 | ≈ 1 个 bbox 边长 |
| **30** | **建议起步**——稍宽松于标准 IoU 0.5 |
| 50 | ≈ 2 个 bbox 边长 |
| 100+ | 实际等于"只要预测到图就命中" |

**怎么选**：
- 看 FN 中"近但不够贴"的 case 占比
- 你的数据：dist_thr=20 → 30 时 FN 从 24 → 22（救了 2 个 TP），再往上没变化
- 说明你的模型 top1 距离要么 < 30px（命中）要么 >> 30px（明显跑偏），**30 是天然甜点**

---

## 3. 怎么跑实验

### 3.1 单次评估（4 口径学术 mAP + 部署口径）

```bash
cd /home/pi/projects/hyperyolo && \
  /home/pi/anaconda3/envs/hyper-yolo/bin/python scripts/lenient_eval.py \
  --weights repos/Hyper-YOLO/runs/coil_v3_rect_imgsz1024_unfrozen_augv2/weights/last.pt \
  --val_dir data/coil/images/val \
  --gt_dir data/coil/labels/val \
  --imgsz 1024 \
  --dist_thresh_nms 30 \
  --dist_thresh_eval 50 \
  --top1_conf_thresh 0.05
```

输出两张表（PR-curves 学术 + top1 部署）：

```
[学术 mAP]
T0 标准 IoU-NMS + IoU-mAP                0.4782  R=0.585  P=0.373
T1 IoU-NMS + Lenient-mAP                 0.6090  R=0.692  P=0.441   ← +13 pp
T2 Soft-NMS + IoU-mAP                    0.4770
T3 Soft-NMS + Lenient-mAP                0.6076

[部署口径]
N0/N2 IoU-Match                          TP=37  FP=21  FN=28  R=0.569  F1=0.602
N1/N3 Lenient-Match                      TP=43  FP=21  FN=22  R=0.662  F1=0.667   ← +6.5 pp
```

注意 N0=N2、N1=N3：NMS 在这条部署链路上对结果零贡献。

### 3.2 网格扫描（找最优工作点）

支持 3 种匹配模式同时扫：`lenient / mac / combined`。

```bash
/home/pi/anaconda3/envs/hyper-yolo/bin/python scripts/scan_top1_thresholds.py \
  --weights repos/Hyper-YOLO/runs/coil_v3_rect_imgsz1024_unfrozen_augv2/weights/last.pt \
  --imgsz 1024 \
  --scan_modes lenient,mac,combined \
  --conf_list 0.001,0.05,0.1,0.15,0.2,0.3,0.5,0.7 \
  --dist_list 20,30,50,80,120 \
  --mac_list 0.3,0.5,0.7 \
  --out_md docs/scan_top1_thresholds_lastpt.md
```

输出 3 张表 + 自动找每模式的最佳工作点 + 写 markdown 报告。

### 3.3 sweep 单参数

```bash
# sweep conf_thresh（保持 dist=30）
for c in 0.001 0.05 0.1 0.2 0.3 0.5 0.7; do
  echo "===== conf=$c ====="
  /home/pi/anaconda3/envs/hyper-yolo/bin/python scripts/lenient_eval.py \
    --weights repos/Hyper-YOLO/runs/coil_v3_rect_imgsz1024_unfrozen_augv2/weights/last.pt \
    --imgsz 1024 --dist_thresh_eval 30 --top1_conf_thresh $c 2>&1 | tail -8
done
```

---

## 4. 怎么读懂结果表

```
================================================================================
[部署口径] Per-image top1（每张图最多 1 个目标，conf_thresh=0.05）
================================================================================
模式                                         TP   FP   FN   TN   Recall  Precision       F1 top1_conf_TP top1_conf_FP
N1: IoU-NMS + Lenient-Match                  36    1   29   63   0.5538     0.9730   0.7059       0.6733       0.1270
```

| 数字 | 含义 | 你应该关注 |
|---|---|---|
| **TP=36** | 真目标里被认出的数量 | 越高越好 |
| **FP=1** | 误报（无 GT 但 top1 命中） | conf_thr 升 → 应该降到 1~3 |
| **FN=29** | 漏检 | dist_thr 升 → 应该降（救回近距 GT） |
| **TN=63** | 正确静默（无 GT 且 top1 被 conf 过滤） | conf_thr 升 → 应该升 |
| **Recall** | TP / (TP+FN) | 漏报代价高 → 关注 |
| **Precision** | TP / (TP+FP) | 误报代价高 → 关注 |
| **F1** | 综合 | 综合比较 |
| **top1_conf_TP** | 真目标 top1 平均 conf | conf 校准好 → 应该远高于 top1_conf_FP |
| **top1_conf_FP** | 误报 top1 平均 conf | conf_thr 升 → 这一项应该被剔除到 FP=0 |

**典型陷阱**：
- **FP=0 但 Recall 也低**：conf_thr 太高，把真目标的低 conf 也过滤了
- **FP 一直高**：conf 校准差，单纯靠 conf_thr 救不回来（要回到训练阶段）
- **dist_thr 怎么调 FN 都不降**：说明 top1 离 GT 太远（>120px），是召回能力问题，不是评估口径问题

---

## 5. 你当前数据集的扫描结果（last.pt，2026-07-04）

### 5.1 完整网格（3 种匹配模式）

#### Lenient（中心距离）

| conf_thr | dist_thr | TP | FP | FN | TN | Recall | Precision | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 20 | 41 | 21 | 24 | 43 | 0.6308 | 0.6613 | 0.6457 |
| 0.001 | 30 | 43 | 21 | 22 | 43 | 0.6615 | 0.6719 | 0.6667 |
| 0.001 | 50+ | 43 | 21 | 22 | 43 | 0.6615 | 0.6719 | 0.6667 |
| **0.05** | **30** | **36** | **1** | **29** | **63** | **0.5538** | **0.9730** | **0.7059** |
| 0.1 | 30 | 36 | 1 | 29 | 63 | 0.5538 | 0.9730 | 0.7059 |
| 0.2 | 30 | 35 | 1 | 30 | 63 | 0.5385 | 0.9722 | 0.6931 |
| 0.5 | 30 | 32 | 1 | 33 | 63 | 0.4923 | 0.9697 | 0.6531 |
| 0.7 | 30 | 30 | 1 | 35 | 63 | 0.4615 | 0.9677 | 0.6250 |

#### MAC（Min-Area Coverage）

| conf_thr | mac_thr | TP | FP | FN | TN | Recall | Precision | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.05** | **0.3** | **36** | **1** | **29** | **63** | **0.5538** | **0.9730** | **0.7059** |
| 0.05 | 0.5 | 36 | 1 | 29 | 63 | 0.5538 | 0.9730 | 0.7059 |
| 0.05 | 0.7 | 35 | 1 | 30 | 63 | 0.5385 | 0.9722 | 0.6931 |
| 0.1 | 0.3 | 36 | 1 | 29 | 63 | 0.5538 | 0.9730 | 0.7059 |
| 0.1 | 0.5 | 36 | 1 | 29 | 63 | 0.5538 | 0.9730 | 0.7059 |
| 0.1 | 0.7 | 35 | 1 | 30 | 63 | 0.5385 | 0.9722 | 0.6931 |

#### Combined（Lenient OR MAC）

| conf_thr | dist_thr | mac_thr | TP | FP | FN | TN | Recall | Precision | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 20~120 | 0.3~0.7 | 43 | 21 | 22 | 43 | 0.6615 | 0.6719 | 0.6667 |
| **0.05** | **20** | **0.3** | **36** | **1** | **29** | **63** | **0.5538** | **0.9730** | **0.7059** |
| 0.05 | 30+ | 0.3+ | 36 | 1 | 29 | 63 | 0.5538 | 0.9730 | 0.7059 |

#### 对照（IoU ≥ 0.5，conf_thr=0.001）

| TP | FP | FN | TN | Recall | Precision | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 37 | 21 | 28 | 43 | 0.5692 | 0.6379 | **0.6016** |

### 5.2 关键解读（3 模式对比）

1. **三种模式收敛到同一 F1 上限 0.7059**：
   - Lenient：conf=0.05, dist=30 → F1=0.7059
   - MAC：conf=0.05, mac=0.3 → F1=0.7059
   - Combined：conf=0.05, dist=20+, mac=0.3+ → F1=0.7059
   - 这不是 bug，是数据特点。

2. **为什么 MAC ≈ Lenient 救回同样的 TP**：
   - 所有"被 Lenient 救回的 TP"（top1 距离 < 30px 但 IoU < 0.5）
   - 在你的数据里这些 case 的 MAC 也 ≥ 0.5（说明预测框虽然位置偏一点，但和 GT 的重叠度足够高）
   - → Lenient 和 MAC 在你这是"同一组 case 的两种描述方式"

3. **Combined 模式在 conf=0.05 时没有超越 Lenient/MAC**：
   - 说明你的数据中不存在"中心点偏离但 MAC 高"的 case
   - → Combined 是"保险"，当数据变化时它能捕获单一指标漏掉的 case，但当前不贡献新收益

4. **conf=0.05 是真正的甜点（所有模式都成立）**：
   - conf=0.05 时 FP 降到 1（conf 校准好）
   - conf 再低 FP 暴涨，再高 TP 也跟着掉

5. **业务最佳工作点（3 模式任选其一）**：
   - `conf_thr=0.05, dist_thr=30 (Lenient)` → F1=0.7059
   - `conf_thr=0.05, mac_thr=0.3 (MAC)` → F1=0.7059
   - `conf_thr=0.05, dist_thr=20, mac_thr=0.3 (Combined)` → F1=0.7059

### 5.3 离最优还有多远

| 现状 | 上限（理论） | 差距 | 来源 |
|---|---|---|---|
| Recall=0.554 | Recall=1.0 | -0.446 | 模型漏检 29 张（conf_thr=0.05 之下） |
| Precision=0.973 | Precision=1.0 | -0.027 | 仅 1 个 FP，提升空间小 |
| F1=0.7059 | F1=1.0 | -0.294 | 主要受 Recall 拖累 |

**结论（最重要的判断）**：

- **评估口径已经达到上限**：3 种匹配模式在任何 conf×thresh 组合下都收敛到同一 F1=0.7059
- **Precision 已接近上限**：97.3%，提升空间仅 2.7 pp
- **Recall 是主要瓶颈**：29 个 FN 是模型召回能力的根本限制，**任何评估口径的微调都救不回来**
- **下一步应该回到训练**：要么加数据（FN 集中的图片），要么换 anchor / loss 看能不能召回

---

## 6. 想扩展怎么做

### 加新评估口径（比如 Center-in-Other，CIO）

笔记里你提过"只要预测框中心落在 GT 内就算 TP"。当前代码已实现 `mac` 和 `combined` 模式，扩展 CIO 可在 `eval_top1_one` 加分支：

```python
elif mode == 'center_in':
    for g in img_gts:
        cx = (top1[0] + top1[2]) / 2
        cy = (top1[1] + top1[3]) / 2
        if g[0] <= cx <= g[2] and g[1] <= cy <= g[3]:
            return 'tp'
    return ('fp' if not img_gts else 'fn')
```

CIO 比 Lenient 还宽容（不看距离，只看中心是否落入框）。在你的数据上可以试，看能不能救回更多 FN。

### 为什么 MAC 和 Lenient 在你数据上等价

观察到的事实：所有被 Lenient 救回的 TP，MAC 也 ≥ 0.5。这不是巧合，是因为：

- 钢卷头部的预测框紧贴目标（pred bbox 通常 20~40 像素边长）
- GT 标注通常包住整个目标（GT bbox 50~80 像素边长）
- 两者即使中心偏移一点，重叠面积仍然接近较小框面积 → MAC ≈ 1.0
- 同时中心距离也 < 30px → Lenient 也命中

如果你将来换数据集或标注策略变化（GT 变得很大或 pred 变得很大），这两个指标就可能分叉。建议：

- 默认用 Lenient（简单直观）
- 同时报告 MAC 作为交叉验证
- 两者结果一致 → 评估可靠
- 两者分叉 → 人工 review 这部分差异 case，看哪个指标更贴合业务

### 集成进 ultralytics validator（侵入式改动）

把 `eval_top1_deploy` 塞进 ultralytics 的 `DetectionValidator.stats`，训练期间每个 epoch 自动出 F1。但这是改 ultralytics 源码——升级 ultralytics 时容易冲突，**不建议**。

更稳的做法：单独脚本批处理（如本教程用的），训练完了再跑。

---

## 7. 总结：创新点 6 的"性价比"

| 评估点 | 结果 |
|---|---|
| 代码行数 | < 150 行（`lenient_eval.py` 里的 `eval_top1_deploy` + `collect_preds_top1`） |
| 训练成本 | **0（纯后处理 / 评估口径）** |
| GPU 推理成本 | 1 次（其余都是 CPU 后处理） |
| 实验时间 | ~1 分钟跑完单次评估；~1 分钟跑完网格扫描 |
| 风险 | 低（不动 ultralytics 源码） |
| 复用价值 | 高（之后所有新 weight 都可以用同一脚本跑部署口径） |
| 你的业务收益 | F1 +10.4 pp（IoU 对照 → Lenient + conf_thr=0.05） |

> 这是笔记 §1.6 里**今天 ROI 最高的创新点** —— 你能用最低成本直接量化"训练指标"和"业务感知精度"之间的差距。

---

## 附录 A：归档声明（创新点 5）

**Soft-NMS-Cov（创新点 5）在本场景下无贡献**，归档说明：

1. **机制层面**：max_det=1 之后 NMS 不再选谁进谁出，IoU-NMS 和 Soft-NMS-Cov 选出的 top1 完全相同。
2. **数据验证**：扫描里 N0=N2、N1=N3 是铁证。
3. **代码层**：`collect_preds_top1` 仍调用 `soft_nms_coverage`，是为了保持评估框架的完整性。**部署时这条调用结果是 no-op**，不需要单独说明。

如果你将来切换到"每图多目标"场景（比如同时检测头部+尾部，或者换成其他数据集），重新启用 Soft-NMS-Cov 只需：
- `lenient_eval.py` 的 NMS 切换保留 `'iou'` / `'soft_coverage'` 两个选项
- 训练时仍然可以跑 N2/N3 对照
- 但**当前钢卷头部场景不用关心这条**

---

## 附录 B：单条命令速查

```bash
# 1. 单次评估（学术 + 部署，4 模式：IoU/Lenient/MAC/Combined）
python scripts/lenient_eval.py \
  --weights <last.pt> --imgsz 1024 \
  --dist_thresh_eval 30 --mac_thresh 0.5 \
  --top1_conf_thresh 0.05

# 2. 网格扫描（3 模式找最优工作点）
python scripts/scan_top1_thresholds.py \
  --weights <last.pt> --imgsz 1024 \
  --scan_modes lenient,mac,combined \
  --conf_list 0.001,0.05,0.1,0.2,0.5,0.7 \
  --dist_list 20,30,50,80,120 \
  --mac_list 0.3,0.5,0.7

# 3. 只看 PR-curves 学术口径
python scripts/lenient_eval.py --weights <last.pt> --mode pr

# 4. 只看 top1 部署口径
python scripts/lenient_eval.py --weights <last.pt> --mode top1 \
  --dist_thresh_eval 30 --mac_thresh 0.5 --top1_conf_thresh 0.05
```