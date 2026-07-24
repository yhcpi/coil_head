"""Bench v41 = v40 + GPU post-pruning. Vary min_branch."""
import sys
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v39
import v41

here = Path(__file__).resolve().parent
project = here.parent
MASKS_DIR = project / "yhc" / "mask_refine"


def main():
    files = sorted(MASKS_DIR.glob("*.png"))
    img0 = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    m0 = (img0 > 127).astype(np.uint8)
    for mb in (5, 8, 12):
        _ = v41.fit_v41(m0, min_branch=mb)
    torch.cuda.synchronize()
    rows_by_mb = {mb: [] for mb in (5, 8, 12)}
    print(f"v41 prune sweep on {len(files)} images...")
    for i, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        mask_bin = (img > 127).astype(np.uint8)
        for mb in (5, 8, 12):
            r = v41.fit_v41(mask_bin, min_branch=mb)
            torch.cuda.synchronize()
            rows_by_mb[mb].append({
                "file": f.name,
                "N": r["N"],
                "t_total": r["elapsed_s"]["total"] * 1000,
                "t_zs": r["elapsed_s"]["zs"] * 1000,
                "t_prune": r["elapsed_s"]["prune"] * 1000,
            })
        if (i+1) % 25 == 0:
            print(f"  {i+1}/{len(files)}")
    print(f"\n========== v41 prune sweep (100 images) ==========")
    print(f"v39 baseline: N_med=4  t_med=108ms")
    print(f"\n{'min_branch':>12} {'N_med':>6} {'N_mean':>7} {'t_med':>7} "
          f"{'zs':>5} {'prune':>6} {'<150ms':>7}")
    for mb in (5, 8, 12):
        rs = rows_by_mb[mb]
        ns = np.array([r["N"] for r in rs])
        ts = np.array([r["t_total"] for r in rs])
        zs = np.array([r["t_zs"] for r in rs])
        ps = np.array([r["t_prune"] for r in rs])
        rt150 = (ts <= 150).mean() * 100
        print(f"{mb:>12} {np.median(ns):>6.0f} {ns.mean():>7.1f} "
              f"{np.median(ts):>7.0f} {np.median(zs):>5.0f} {np.median(ps):>6.1f} "
              f"{rt150:>6.0f}%")
    # Gold
    g = cv2.imread(str(project / "yhc/mask_refine/017_roi_004.png"),
                   cv2.IMREAD_GRAYSCALE)
    m = (g > 127).astype(np.uint8)
    print(f"\n[GOLD 017_roi_004] GT=3")
    for mb in (5, 8, 12):
        r = v41.fit_v41(m, min_branch=mb)
        print(f"  mb={mb}: N={r['N']}")


if __name__ == "__main__":
    main()
