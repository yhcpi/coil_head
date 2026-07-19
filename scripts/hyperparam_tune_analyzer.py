#!/usr/bin/env python3
"""2026-07-15 给定训练 run 的 results.csv，自动分析失败 / 波动 / 正常 并给出建议。

用法：
  python scripts/hyperparam_tune_analyzer.py --name v0_dy_hyper_yolon_250ep

输出：
  - stdout：分类结果 + 关键指标摘要
  - runs/baseline/<NAME>/tune_report.md

判定阈值（基于项目历史）：
  max mAP50 < 0.5        → FAIL：列出可能根因
  0.5 ≤ max mAP50 < 0.85 → OSCILLATING：调优建议
  max mAP50 ≥ 0.85       → NORMAL：上界优化建议（TTA / conf scan / hard neg crop）
"""
import argparse
import csv
import math
import statistics
from pathlib import Path

REPO = Path("/home/pi/projects/hyperyolo")

# 列定义（ultralytics 8.x results.csv 真实列序，详见 ultracode_truth_2026-07-11）
COL_EPOCH = 0
COL_BOX = 1
COL_CLS = 2
COL_DFL = 3
COL_P = 7
COL_R = 8
COL_MAP50 = 9
COL_MAP50_95 = 10
COL_VAL_BOX = 11


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="runs/baseline/<NAME> 的 NAME")
    p.add_argument("--runs-dir", default=str(REPO / "runs/baseline"))
    p.add_argument("--fail-thr", type=float, default=0.5)
    p.add_argument("--osc-thr", type=float, default=0.85)
    return p.parse_args()


def read_results(csv_path: Path):
    """返回 list[dict]，每行一个 epoch。空文件返回 []。"""
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path) as f:
        rdr = csv.reader(f)
        header = None
        for row in rdr:
            if not row or row[0].strip().startswith("epoch"):
                header = row
                continue
            try:
                epoch = int(row[COL_EPOCH].strip())
                rows.append({
                    "epoch": epoch,
                    "box": float(row[COL_BOX]),
                    "cls": float(row[COL_CLS]),
                    "dfl": float(row[COL_DFL]),
                    "P": float(row[COL_P]),
                    "R": float(row[COL_R]),
                    "mAP50": float(row[COL_MAP50]),
                    "mAP50_95": float(row[COL_MAP50_95]),
                    "val_box": float(row[COL_VAL_BOX]),
                })
            except (ValueError, IndexError):
                continue
    return rows


def load_args_yaml(run_dir: Path):
    """读取 run_dir/args.yaml 拿训练参数。"""
    p = run_dir / "args.yaml"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def analyze_failure(rows, args_yaml, name):
    """max mAP50 < 0.5：列出可能根因。"""
    causes = []
    checks = []

    last = rows[-1] if rows else {}
    box = last.get("box", float("nan"))
    cls = last.get("cls", float("nan"))
    val_box = last.get("val_box", float("nan"))
    R = last.get("R", 0)
    P = last.get("P", 0)
    max_map = max((r["mAP50"] for r in rows), default=0)
    final_map = last.get("mAP50", 0)
    box_decay = rows[0]["box"] - rows[-1]["box"] if rows else 0

    # 1. cls_remap 单类问题
    nc = args_yaml.get("nc") or args_yaml.get("names")
    if nc in ("1", "['coil_head']", "coil_head"):
        checks.append(("- 数据集类别数", f"{nc}（单类 — 正常，检查 cls_loss 是否持续下降）"))

    # 2. box=7.5 vs 1.5 失配
    box_w = args_yaml.get("box", "?")
    if box_w in ("7.5", "5.0"):
        causes.append(f"**box loss 权重 = {box_w}**：过重 → 压制 cls/df1 学习。建议改回 1.5（v8 baseline 标准）。")
    elif box_w in ("1.5", "2.0"):
        checks.append(("- box loss 权重", f"{box_w}（标准值）"))
    else:
        checks.append(("- box loss 权重", f"{box_w}（非标准）"))

    # 3. nwd 缺失
    nwd = args_yaml.get("nwd", "?")
    if nwd in ("false", "False", "0"):
        causes.append("**NWD 未启用**（nwd=false）：小目标 IoU 梯度噪声大，NWD 是钢卷 tip 场景的标准配置。建议 nwd=true, nwd_constant=12.0。")
    elif nwd in ("true", "True", "1"):
        checks.append(("- NWD", f"{nwd}（已启用）"))

    # 4. pretrain 不匹配
    model = args_yaml.get("model", "?")
    pretrained = args_yaml.get("pretrained", "?")
    checks.append(("- 模型", f"{model} (pretrained={pretrained})"))

    # 5. aug 太重
    degrees = args_yaml.get("degrees", "0")
    flipud = args_yaml.get("flipud", "0")
    copy_paste = args_yaml.get("copy_paste", "0")
    aug_score = 0
    try:
        if float(degrees) >= 10: aug_score += 1
        if float(flipud) >= 0.3: aug_score += 1
        if float(copy_paste) >= 0.1: aug_score += 1
    except ValueError:
        pass
    if aug_score >= 2:
        causes.append(f"**aug 太重**（degrees={degrees}/flipud={flipud}/copy_paste={copy_paste}）：钢卷 tip 小目标对旋转 + 翻转 + 粘贴噪声敏感。建议 weak aug：degrees=0/flipud=0/copy_paste=0（参考 v18.3 部署最优 F1=0.9286）。")
    elif aug_score == 0:
        checks.append(("- aug 强度", f"degrees={degrees}/flipud={flipud}/copy_paste={copy_paste}（弱 aug）"))
    else:
        checks.append(("- aug 强度", f"degrees={degrees}/flipud={flipud}/copy_paste={copy_paste}（中 aug）"))

    # 6. 数据问题
    if final_map < 0.05 and box_decay > 2:
        causes.append("**box_loss 收敛但 mAP50 ≈ 0**：典型数据问题——标签错位 / 单图多目标 / bbox 全零 / 单类未注册。检查 data.yaml 中 nc 与 labels 实际 cls id 是否一致。")

    # 7. mAP50 从未上升
    if max_map < 0.1 and len(rows) > 5:
        causes.append(f"**mAP50 在 {len(rows)} epoch 内始终 < 0.1**：很可能是 coverage loss 实现 bug 或 yaml 加载失败。检查 src/hyper_yolo_patches/PATCHES.md 是否生效。")

    # 8. recall = 0
    if R == 0 and P == 0 and len(rows) > 10:
        causes.append("**P=R=0**：模型从未产生预测。检查 conf 阈值或 max_det 是否被 patch 卡死。")

    return {
        "category": "FAIL",
        "summary": f"max mAP50 = {max_map:.4f}, final = {final_map:.4f}, "
                   f"box {rows[0]['box']:.2f}→{box:.2f} (Δ={box_decay:.2f}), R={R:.3f}",
        "checks": checks,
        "causes": causes,
    }


def analyze_oscillating(rows, args_yaml, max_map):
    """0.5 ≤ max mAP50 < 0.85：调优建议。"""
    suggestions = []
    checks = []

    last = rows[-1]
    final_map = last["mAP50"]

    # 收敛性：最后 10 epoch 波动
    tail = [r["mAP50"] for r in rows[-10:]] if len(rows) >= 10 else [r["mAP50"] for r in rows]
    if len(tail) >= 3:
        std = statistics.pstdev(tail)
        rng = max(tail) - min(tail)
    else:
        std = rng = 0

    # 1. patience
    patience = args_yaml.get("patience", "?")
    checks.append(f"- patience = {patience}")
    if patience in ("50", "30"):
        suggestions.append(f"**patience={patience} 可能过早停**：本 run max={max_map:.4f} 但 final={final_map:.4f}，"
                           f"差 {max_map - final_map:.4f}。考虑 patience=80 或 100 给余量。")

    # 2. lr
    lr0 = args_yaml.get("lr0", "?")
    checks.append(f"- lr0 = {lr0}")
    if lr0 in ("0.01", "0.02"):
        suggestions.append(f"**lr0={lr0} 偏高**：250 epoch 余弦退火到 lrf=0.01 但起步冲击大。"
                           f"对低数据量 (train=312) 场景建议 lr0=0.005 + warmup_epochs=5。")

    # 3. aug 强度
    degrees = args_yaml.get("degrees", "0")
    flipud = args_yaml.get("flipud", "0")
    copy_paste = args_yaml.get("copy_paste", "0")
    checks.append(f"- aug: degrees={degrees}/flipud={flipud}/copy_paste={copy_paste}")
    if float(degrees or 0) >= 10 or float(flipud or 0) >= 0.3:
        suggestions.append("**aug 中-强**：考虑 weak aug（degrees=0/flipud=0/copy_paste=0）"
                           "——v18.3 验证弱 aug 在小目标场景稳定赢强 aug +5pp。")

    # 4. NWD 权重
    nwd = args_yaml.get("nwd", "?")
    nwd_c = args_yaml.get("nwd_constant", "?")
    checks.append(f"- NWD: nwd={nwd}, constant={nwd_c}")
    if nwd == "true":
        suggestions.append(f"**NWD constant = {nwd_c}**：若 max mAP50 仍未到 0.85，"
                           "考虑 nwd_constant=8.0（更紧凑高斯，匹配小目标）。")

    # 5. box 权重
    box_w = args_yaml.get("box", "?")
    checks.append(f"- box loss 权重 = {box_w}")

    # 6. 波动大
    if std > 0.05 and rng > 0.1:
        suggestions.append(f"**末 10 epoch 波动 std={std:.3f} range={rng:.3f}**：不稳定。"
                           "考虑关闭强 aug + 加大 batch + 启用 EMA（patch 默认开启）已开就排除。")

    return {
        "category": "OSCILLATING",
        "summary": f"max mAP50 = {max_map:.4f}, final = {final_map:.4f}, "
                   f"tail std={std:.3f}, range={rng:.3f}",
        "checks": checks,
        "causes": suggestions,
    }


def analyze_normal(rows, args_yaml, max_map):
    """max mAP50 ≥ 0.85：上界优化建议。"""
    suggestions = []
    checks = []

    last = rows[-1]

    # 1. TTA
    suggestions.append("**TTA-builtin**：model.predict(augment=True) — 验证 +2-5pp（v18.3 验证）。")

    # 2. conf sweep 部署 F1
    suggestions.append("**conf sweep 部署 F1**：用 scripts/scan_top1_thresholds.py 或 eval_c_d_runs.py "
                       "扫 conf ∈ {0.05, 0.10, 0.15, 0.20} + dist=30 → 找最佳部署 F1（目标 ≥ 0.9286）。")

    # 3. hard neg crop
    if "v18" not in args_yaml.get("name", ""):
        suggestions.append("**hard neg crop**（v18.3 验证 +1.50pp F1）：用 scripts/hard_neg_mine.py 找出 FN 原图，"
                           "× 3 副本 + **弱 aug + lr=0.005 + 100ep** 闭环训练。")

    # 4. 蒸馏 / model ensemble（仅 mAP50 ≥ 0.88 时考虑）
    if max_map >= 0.88:
        suggestions.append("**max ≥ 0.88 接近 v12 学术顶**（0.882）：可考虑 model ensemble / "
                           "scale-aware TTA（多 imgsz 推理投票）。")

    checks.append(f"- max mAP50 = {max_map:.4f}")
    checks.append(f"- final mAP50 = {last['mAP50']:.4f}")
    checks.append(f"- 末 epoch P={last['P']:.3f} R={last['R']:.3f}")

    return {
        "category": "NORMAL",
        "summary": f"max mAP50 = {max_map:.4f}, final = {last['mAP50']:.4f}",
        "checks": checks,
        "causes": suggestions,
    }


def render_markdown(name, rows, args_yaml, analysis):
    md = []
    md.append(f"# Hyperparam Tune Report — `{name}`\n")
    md.append(f"> 分类: **{analysis['category']}**\n")
    md.append(f"> {analysis['summary']}\n")
    md.append("")
    md.append("## 检查项\n")
    for c in analysis["checks"]:
        if isinstance(c, tuple):
            md.append(f"- **{c[0]}**: {c[1]}")
        else:
            md.append(c)
    md.append("")
    if analysis["causes"]:
        md.append(f"## {'可能根因' if analysis['category'] == 'FAIL' else '建议'}\n")
        for i, c in enumerate(analysis["causes"], 1):
            md.append(f"{i}. {c}")
        md.append("")
    # 关键指标
    md.append("## 关键指标\n")
    md.append("| epoch | box | cls | P | R | mAP50 | mAP50-95 |")
    md.append("|-------|-----|-----|---|---|-------|----------|")
    for r in rows[-10:]:
        md.append(f"| {r['epoch']} | {r['box']:.3f} | {r['cls']:.3f} | "
                  f"{r['P']:.3f} | {r['R']:.3f} | {r['mAP50']:.4f} | {r['mAP50_95']:.4f} |")
    md.append("")
    if rows:
        max_idx = max(range(len(rows)), key=lambda i: rows[i]["mAP50"])
        md.append(f"- **max mAP50 epoch**: {rows[max_idx]['epoch']} (={rows[max_idx]['mAP50']:.4f})")
    md.append("")
    return "\n".join(md)


def main():
    args = parse_args()
    run_dir = Path(args.runs_dir) / args.name
    csv_path = run_dir / "results.csv"

    print(f"[analyzer] {run_dir}")
    rows = read_results(csv_path)
    if not rows:
        print(f"[analyzer] results.csv 不存在或为空: {csv_path}")
        return

    max_map = max(r["mAP50"] for r in rows)
    print(f"[analyzer] 共 {len(rows)} epoch, max mAP50 = {max_map:.4f}")

    args_yaml = load_args_yaml(run_dir)

    if max_map < args.fail_thr:
        analysis = analyze_failure(rows, args_yaml, args.name)
    elif max_map < args.osc_thr:
        analysis = analyze_oscillating(rows, args_yaml, max_map)
    else:
        analysis = analyze_normal(rows, args_yaml, max_map)

    print(f"\n[{analysis['category']}] {analysis['summary']}\n")
    print("检查项:")
    for c in analysis["checks"]:
        print(f"  {c}")
    if analysis["causes"]:
        print(f"\n{'可能根因' if analysis['category']=='FAIL' else '建议'}:")
        for i, c in enumerate(analysis["causes"], 1):
            print(f"  {i}. {c}")

    # 写 markdown
    out_path = run_dir / "tune_report.md"
    out_path.write_text(render_markdown(args.name, rows, args_yaml, analysis))
    print(f"\n报告已写入: {out_path}")


if __name__ == "__main__":
    main()