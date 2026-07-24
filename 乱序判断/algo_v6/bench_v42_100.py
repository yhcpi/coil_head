"""Bench v42 (DT medial axis) vs v39 (skimage Lee) on 100 images."""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v39
import v42

here = Path(__file__).resolve().parent
project = here.parent
MASKS_DIR = project / "yhc" / "mask_refine"


def main():
    files = sorted(MASKS_DIR.glob("*.png"))
    img0 = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    m0 = (img0 > 127).astype(np.uint8)
    _ = v39.fit_v39(m0); _ = v42.fit_v42(m0)
    torch.cuda.synchronize()

    rows = []
    print(f"v39 (skimage Lee) vs v42 (cv2 DT) on {len(files)} images...")
    for i, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        mask_bin = (img > 127).astype(np.uint8)
        r39_ = v39.fit_v39(mask_bin); torch.cuda.synchronize()
        r42_ = v42.fit_v42(mask_bin); torch.cuda.synchronize()
        rows.append({
            "file": f.name,
            "v39_N": r39_["N"], "v39_t": r39_["elapsed_s"]["total"] * 1000,
            "v39_skel_t": r39_["elapsed_s"]["skel"] * 1000,
            "v42_N": r42_["N"], "v42_t": r42_["elapsed_s"]["total"] * 1000,
            "v42_skel_t": r42_["elapsed_s"]["skel"] * 1000,
        })
        if (i+1) % 25 == 0:
            print(f"  {i+1}/{len(files)}")
    n39 = np.array([r["v39_N"] for r in rows])
    n42 = np.array([r["v42_N"] for r in rows])
    t39 = np.array([r["v39_t"] for r in rows])
    t42 = np.array([r["v42_t"] for r in rows])
    skel39 = np.array([r["v39_skel_t"] for r in rows])
    skel42 = np.array([r["v42_skel_t"] for r in rows])
    delta_n = n42 - n39
    print(f"\n========== v39 vs v42 (100 images) ==========")
    print(f"v39:  N_med={np.median(n39):.0f} N_mean={n39.mean():.1f}  "
          f"t_med={np.median(t39):.0f}ms  skel_t_med={np.median(skel39):.0f}ms")
    print(f"v42:  N_med={np.median(n42):.0f} N_mean={n42.mean():.1f}  "
          f"t_med={np.median(t42):.0f}ms  skel_t_med={np.median(skel42):.0f}ms")
    print(f"\ndelta_N (v42 - v39):  med={np.median(delta_n):.1f}  "
          f"min={delta_n.min():.0f} max={delta_n.max():.0f}  "
          f"v42 lower: {(delta_n < 0).sum()}/100  "
          f"|delta|<=1: {(np.abs(delta_n) <= 1).sum()}/100")
    print(f"\nskel speedup (med): {np.median(skel39)/np.median(skel42):.2f}x "
          f"({np.median(skel39):.0f}ms → {np.median(skel42):.0f}ms)")
    print(f"total speedup (med): {np.median(t39)/np.median(t42):.2f}x "
          f"({np.median(t39):.0f}ms → {np.median(t42):.0f}ms)")
    for v_name, ts in [("v39", t39), ("v42", t42)]:
        rt100 = (ts <= 100).mean() * 100
        rt150 = (ts <= 150).mean() * 100
        rt200 = (ts <= 200).mean() * 100
        print(f"{v_name}: <100ms={rt100:.0f}%  <150ms={rt150:.0f}%  <200ms={rt200:.0f}%")
    # Gold standard
    g = cv2.imread(str(project / "yhc/mask_refine/017_roi_004.png"),
                   cv2.IMREAD_GRAYSCALE)
    m = (g > 127).astype(np.uint8)
    r39g = v39.fit_v39(m); r42g = v42.fit_v42(m)
    print(f"\n[GOLD 017_roi_004] v39 N={r39g['N']}  v42 N={r42g['N']}  GT=3")
    import json
    out = project / "algo_v6" / "v42_bench.json"
    out.write_text(json.dumps(rows, indent=2, default=str,
                             ensure_ascii=False))
    print(f"\n[json] {out}")


if __name__ == "__main__":
    main()
