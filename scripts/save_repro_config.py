#!/usr/bin/env python3
"""2026-07-15 部署权重归档：当某个 run 表现超过 v12 baseline 时复制 best.pt + 配置 + commit 元数据。

触发条件（任一满足即可）：
  - 学术 mAP50 > 0.882（v12 baseline，列 9 = metrics/mAP50(B)，新口径 model.val()）
  - 部署 F1 > 0.9286（v18.3 部署最优）

用法：
  python scripts/save_repro_config.py --name v0_dy_hyper_yolon_250ep \
      --headline-mAP50 0.91 --headline-F1 0.93 \
      --description "DySample neck innovation"

归档内容（runs/deploy_best/<NAME>_vN.M/）：
  - best.pt         : 训练产出
  - args.yaml       : 训练参数快照
  - results.csv     : 训练曲线
  - TRAIN_CONFIG.md : 训练配置说明（如果存在）
  - commit_metadata.txt : 提交 hash/作者/日期/msg
  - README.md       : headline 指标 + 描述 + 触发原因
"""
import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path("/home/pi/projects/hyperyolo")
DEPLOY_BEST = REPO / "runs/deploy_best"

# v12 baseline（学术 mAP50，model.val() 新口径）+ v18.3（部署 F1）
THRESHOLD_ACADEMIC_MAP50 = 0.882
THRESHOLD_DEPLOY_F1 = 0.9286


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="runs/baseline/<NAME> 的 NAME")
    p.add_argument("--headline-mAP50", type=float, default=None,
                   help="学术 mAP50（model.val() 新口径，可选）")
    p.add_argument("--headline-F1", type=float, default=None,
                   help="部署 F1（TTA-builtin + dist≤30，可选）")
    p.add_argument("--description", default="", help="模型/创新点描述")
    p.add_argument("--force", action="store_true",
                   help="即使未超过阈值也强制归档")
    p.add_argument("--runs-dir", default=str(REPO / "runs/baseline"))
    p.add_argument("--deploy-dir", default=str(DEPLOY_BEST))
    return p.parse_args()


def git(*args, cwd=REPO):
    """git 子进程封装，失败返回空字符串。"""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd), capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def collect_commit_metadata():
    """收集当前 commit 的元数据。"""
    h = git("rev-parse", "HEAD")
    short = git("rev-parse", "--short", "HEAD")
    author = git("log", "-1", "--pretty=format:%an")
    date = git("log", "-1", "--pretty=format:%ai")
    subject = git("log", "-1", "--pretty=format:%s")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status = git("status", "--short")
    dirty = " (dirty)" if status else ""
    return {
        "hash": h,
        "short": short,
        "author": author,
        "date": date,
        "subject": subject,
        "branch": branch,
        "dirty_status": status,
        "dirty": bool(status),
        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def next_version(name: str, deploy_dir: Path):
    """扫描 deploy_dir 找同名最大 vN.M，返回下一个。"""
    prefix = f"{name}_v"
    used = []
    if deploy_dir.exists():
        for d in deploy_dir.iterdir():
            if d.is_dir() and d.name.startswith(prefix):
                tag = d.name[len(prefix):]
                if "." in tag:
                    try:
                        a, b = tag.split(".", 1)
                        used.append((int(a), int(b)))
                    except ValueError:
                        continue
    if not used:
        return (1, 0)
    used.sort()
    last = used[-1]
    return (last[0], last[1] + 1)


def check_threshold(mAP50, F1, force):
    """判定是否超过 v12 baseline。"""
    reasons = []
    if mAP50 is not None and mAP50 > THRESHOLD_ACADEMIC_MAP50:
        reasons.append(f"学术 mAP50 = {mAP50:.4f} > v12 0.882 (+{mAP50 - THRESHOLD_ACADEMIC_MAP50:.4f})")
    if F1 is not None and F1 > THRESHOLD_DEPLOY_F1:
        reasons.append(f"部署 F1 = {F1:.4f} > v18.3 0.9286 (+{F1 - THRESHOLD_DEPLOY_F1:.4f})")
    return reasons


def copy_files(src_run: Path, dst_dir: Path):
    """从 run_dir 拷贝必要文件到 deploy_best/<name>_vN.M/。"""
    dst_dir.mkdir(parents=True, exist_ok=True)

    # best.pt（必须有）
    best = src_run / "weights/best.pt"
    if not best.exists():
        raise FileNotFoundError(f"{best} 不存在")
    shutil.copy2(best, dst_dir / "best.pt")

    # args.yaml
    args_yaml = src_run / "args.yaml"
    if args_yaml.exists():
        shutil.copy2(args_yaml, dst_dir / "args.yaml")

    # results.csv
    results = src_run / "results.csv"
    if results.exists():
        shutil.copy2(results, dst_dir / "results.csv")

    # TRAIN_CONFIG.md（如果存在）
    train_cfg = src_run / "TRAIN_CONFIG.md"
    if train_cfg.exists():
        shutil.copy2(train_cfg, dst_dir / "TRAIN_CONFIG.md")


def write_commit_metadata(dst_dir: Path, meta: dict):
    path = dst_dir / "commit_metadata.txt"
    lines = [
        f"# Commit Metadata @ {meta['collected_at']}",
        f"hash:        {meta['hash']}",
        f"short:       {meta['short']}",
        f"branch:      {meta['branch']}{' (dirty)' if meta['dirty'] else ''}",
        f"author:      {meta['author']}",
        f"date:        {meta['date']}",
        f"subject:     {meta['subject']}",
    ]
    if meta["dirty_status"]:
        lines.append("")
        lines.append("# Working tree status:")
        for ln in meta["dirty_status"].splitlines():
            lines.append(f"  {ln}")
    path.write_text("\n".join(lines) + "\n")


def write_readme(dst_dir: Path, name: str, mAP50, F1, description, reasons, meta, version):
    path = dst_dir / "README.md"
    lines = [
        f"# Deploy Best: `{name}_v{version[0]}.{version[1]}`",
        "",
        f"> Auto-saved by `scripts/save_repro_config.py` @ {meta['collected_at']}",
        "",
        "## Headline",
        "",
        "| 指标 | 值 | v12 baseline | Δ |",
        "|------|----|--------------|---|",
    ]
    if mAP50 is not None:
        delta = mAP50 - THRESHOLD_ACADEMIC_MAP50
        sign = "+" if delta >= 0 else ""
        lines.append(f"| 学术 mAP50 (model.val conf=0.001 iou=0.6 max_det=1) | **{mAP50:.4f}** | {THRESHOLD_ACADEMIC_MAP50} | {sign}{delta:.4f} |")
    if F1 is not None:
        delta = F1 - THRESHOLD_DEPLOY_F1
        sign = "+" if delta >= 0 else ""
        lines.append(f"| 部署 F1 (TTA-builtin + top-1 + dist≤30) | **{F1:.4f}** | {THRESHOLD_DEPLOY_F1} | {sign}{delta:.4f} |")
    lines.append("")
    if reasons:
        lines.append("## 触发原因")
        for r in reasons:
            lines.append(f"- {r}")
        lines.append("")
    if description:
        lines.append("## 描述")
        lines.append(description)
        lines.append("")
    lines.append("## Commit")
    lines.append(f"- `{meta['short']}` {meta['subject']}")
    lines.append(f"- by {meta['author']} at {meta['date']}")
    lines.append(f"- branch: `{meta['branch']}`{' (dirty)' if meta['dirty'] else ''}")
    lines.append("")
    lines.append("## 文件清单")
    for f in sorted(dst_dir.iterdir()):
        lines.append(f"- `{f.name}`")
    lines.append("")
    lines.append("## 复现命令")
    lines.append("```bash")
    lines.append(f"# 从 commit {meta['short']} checkout")
    lines.append(f"git checkout {meta['short']}")
    lines.append(f"PYTHONPATH= python -m ultralytics.train \\")
    lines.append(f"    model=repos/Hyper-YOLO/hyper-yolon.pt \\")
    lines.append(f"    data=data/coil/data.yaml \\")
    lines.append(f"    imgsz=1024 epochs=250 batch=16 \\")
    lines.append(f"    project=runs/baseline name={name} exist_ok=true")
    lines.append("```")
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    src_run = Path(args.runs_dir) / args.name
    if not src_run.exists():
        print(f"[ERROR] {src_run} 不存在", file=sys.stderr)
        sys.exit(1)

    deploy_dir = Path(args.deploy_dir)
    deploy_dir.mkdir(parents=True, exist_ok=True)

    reasons = check_threshold(args.headline_mAP50, args.headline_F1, args.force)
    if not reasons:
        print("[GATE] 未超过 v12 / v18.3 阈值，未归档。")
        print(f"  学术 mAP50 = {args.headline_mAP50} (阈值 {THRESHOLD_ACADEMIC_MAP50})")
        print(f"  部署 F1    = {args.headline_F1} (阈值 {THRESHOLD_DEPLOY_F1})")
        print(f"  传 --force 强制归档")
        sys.exit(0)

    version = next_version(args.name, deploy_dir)
    dst = deploy_dir / f"{args.name}_v{version[0]}.{version[1]}"
    if dst.exists():
        print(f"[ERROR] {dst} 已存在，手动删除或换 --name")
        sys.exit(1)

    print(f"[archive] {dst}")
    copy_files(src_run, dst)
    meta = collect_commit_metadata()
    write_commit_metadata(dst, meta)
    write_readme(dst, args.name, args.headline_mAP50, args.headline_F1,
                 args.description, reasons, meta, version)
    print(f"[archive] 已归档：{dst}")
    print(f"  - 触发原因：{', '.join(reasons)}")
    print(f"  - commit: {meta['short']} {meta['subject']}")


if __name__ == "__main__":
    main()