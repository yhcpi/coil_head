"""Bench v40 (cupy skeleton) vs v39 (skimage) on 100 images."""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v39
import v40

here = Path(__file__).resolve().parent
project = here.parent
MASKS_DIR = project / "yhc" / "mask_refine"


def main():
    files = sorted(MASKS_DIR.glob("*.png"))
    img0 = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    m0 = (img0 > 127).astype(np.uint8)
    _ = v39.fit_v39(m0); _ = v40.fit_v40(m0)
    torch.cuda.synchronize()

    rows = []
    print(f"v39 (skimage skel) vs v40 (cupy ZS skel) on {len(files)} images...")
    for i, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        mask_bin = (img > 127).astype(np.uint8)
        r39_ = v39.fit_v39(mask_bin); torch.cuda.synchronize()
        r40_ = v40.fit_v40(mask_bin); torch.cuda.synchronize()
        rows.append({
            "file": f.name,
            "v39_N": r39_["N"], "v39_t": r39_["elapsed_s"]["total"] * 1000,
            "v39_skel_t": r39_["elapsed_s"]["skel"] * 1000,
            "v40_N": r40_["N"], "v40_t": r40_["elapsed_s"]["total"] * 1000,
            "v40_skel_t": r40_["elapsed_s"]["skel"] * 1000,
        })
        if (i+1) % 25 == 0:
            print(f"  {i+1}/{len(files)}")
    n39 = np.array([r["v39_N"] for r in rows])
    n40 = np.array([r["v40_N"] for r in rows])
    t39 = np.array([r["v39_t"] for r in rows])
    t40 = np.array([r["v40_t"] for r in rows])
    skel39 = np.array([r["v39_skel_t"] for r in rows])
    skel40 = np.array([r["v40_skel_t"] for r in rows])
    delta_n = n40 - n39
    delta_t = t40 - t39
    print(f"\n========== v39 vs v40 (100 images) ==========")
    print(f"v39:  N_med={np.median(n39):.0f} N_mean={n39.mean():.1f}  "
          f"t_med={np.median(t39):.0f}ms  skel_t_med={np.median(skel39):.0f}ms")
    print(f"v40:  N_med={np.median(n40):.0f} N_mean={n40.mean():.1f}  "
          f"t_med={np.median(t40):.0f}ms  skel_t_med={np.median(skel40):.0f}ms")
    print(f"\ndelta_N (v40 - v39):  med={np.median(delta_n):.1f}  "
          f"min={delta_n.min():.0f} max={delta_n.max():.0f}  "
          f"|delta|<=1: {(np.abs(delta_n) <= 1).sum()}/100")
    print(f"delta_t (v40 - v39):  med={np.median(delta_t):+.1f}ms  "
          f"v40 faster: {(delta_t < 0).sum()}/100")
    skel_speedup = np.median(skel39) / np.median(skel40)
    total_speedup = np.median(t39) / np.median(t40)
    print(f"\nskel speedup (med): {skel_speedup:.2f}x "
          f"({np.median(skel39):.0f}ms → {np.median(skel40):.0f}ms)")
    print(f"total speedup (med): {total_speedup:.2f}x "
          f"({np.median(t39):.0f}ms → {np.median(t40):.0f}ms)")
    # Real-time fractions
    for v_name, ts in [("v39", t39), ("v40", t40)]:
        rt100 = (ts <= 100).mean() * 100
        rt150 = (ts <= 150).mean() * 100
        rt200 = (ts <= 200).mean() * 100
        print(f"{v_name}: <100ms={rt100:.0f}%  <150ms={rt150:.0f}%  <200ms={rt200:.0f}%")
    # Gold standard
    g = cv2.imread(str(project / "yhc/mask_refine/017_roi_004.png"),
                   cv2.IMREAD_GRAYSCALE)
    m = (g > 127).astype(np.uint8)
    r39g = v39.fit_v39(m); r40g = v40.fit_v40(m)
    print(f"\n[GOLD 017_roi_004] v39 N={r39g['N']}  v40 N={r40g['N']}  GT=3")
    import json
    out = project / "algo_v6" / "v40_bench.json"
    out.write_text(json.dumps(rows, indent=2, default=str,
                             ensure_ascii=False))
    print(f"\n[json] {out}")


if __name__ == "__main__":
    main()
