"""Bench v44 (Halir-Flusser refit) vs v39 (cv2 fitEllipse) on 100 images."""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v39
import v44

here = Path(__file__).resolve().parent
project = here.parent
MASKS_DIR = project / "yhc" / "mask_refine"


def main():
    files = sorted(MASKS_DIR.glob("*.png"))
    img0 = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    m0 = (img0 > 127).astype(np.uint8)
    _ = v39.fit_v39(m0); _ = v44.fit_v44(m0)
    torch.cuda.synchronize()

    rows = []
    print(f"v39 (cv2 fitEllipse) vs v44 (Halir-Flusser) on {len(files)} images...")
    for i, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        mask_bin = (img > 127).astype(np.uint8)
        r39_ = v39.fit_v39(mask_bin); torch.cuda.synchronize()
        r44_ = v44.fit_v44(mask_bin); torch.cuda.synchronize()
        # Per-circle a/b diff (matching by r within 5px)
        diffs = []
        for c39 in r39_["circles"]:
            best = None; best_d = 1e9
            for c44 in r44_["circles"]:
                d = abs(c39["r"] - c44["r"])
                if d < best_d and d < 5:
                    best_d = d; best = c44
            if best is not None:
                a_diff = abs(c39["a"] - best["a"])
                b_diff = abs(c39["b"] - best["b"])
                diffs.append({"file": f.name, "r": c39["r"],
                              "a39": c39["a"], "a44": best["a"], "a_diff": a_diff,
                              "b39": c39["b"], "b44": best["b"], "b_diff": b_diff})
        rows.append({"file": f.name,
                     "v39_N": r39_["N"], "v39_t": r39_["elapsed_s"]["total"] * 1000,
                     "v44_N": r44_["N"], "v44_t": r44_["elapsed_s"]["total"] * 1000,
                     "diffs": diffs})
        if (i+1) % 25 == 0:
            print(f"  {i+1}/{len(files)}")
    n39 = np.array([r["v39_N"] for r in rows])
    n44 = np.array([r["v44_N"] for r in rows])
    t39 = np.array([r["v39_t"] for r in rows])
    t44 = np.array([r["v44_t"] for r in rows])
    print(f"\n========== v39 vs v44 (100 images) ==========")
    print(f"v39:  N_med={np.median(n39):.0f} N_mean={n39.mean():.1f}  "
          f"t_med={np.median(t39):.0f}ms")
    print(f"v44:  N_med={np.median(n44):.0f} N_mean={n44.mean():.1f}  "
          f"t_med={np.median(t44):.0f}ms")
    print(f"|delta_N|<=1: {(np.abs(n44 - n39) <= 1).sum()}/100")
    # a/b diff
    all_diffs = [d for r in rows for d in r["diffs"]]
    if all_diffs:
        a_diffs = np.array([d["a_diff"] for d in all_diffs])
        b_diffs = np.array([d["b_diff"] for d in all_diffs])
        print(f"\na-axis diff (per matched circle):")
        print(f"  med {np.median(a_diffs):.3f}px  mean {a_diffs.mean():.3f}px  "
              f"p95 {np.percentile(a_diffs, 95):.3f}px  max {a_diffs.max():.3f}px")
        print(f"b-axis diff (per matched circle):")
        print(f"  med {np.median(b_diffs):.3f}px  mean {b_diffs.mean():.3f}px  "
              f"p95 {np.percentile(b_diffs, 95):.3f}px  max {b_diffs.max():.3f}px")
        print(f"# matched circles total: {len(all_diffs)}")
        # HF vs cv2 axis swap check
        swapped = sum(1 for d in all_diffs
                      if (d["a39"] > d["b39"] and d["a44"] < d["b44"]) or
                         (d["a39"] < d["b39"] and d["a44"] > d["b44"]))
        print(f"# axis-swap vs cv2: {swapped}/{len(all_diffs)}")
    # Gold standard
    g = cv2.imread(str(project / "yhc/mask_refine/017_roi_004.png"),
                   cv2.IMREAD_GRAYSCALE)
    m = (g > 127).astype(np.uint8)
    print(f"\n[GOLD 017_004] v39 N={v39.fit_v39(m)['N']}  v44 N={v44.fit_v44(m)['N']}  GT=3")
    import json
    out = project / "algo_v6" / "v44_bench.json"
    out.write_text(json.dumps(rows, indent=2, default=str,
                             ensure_ascii=False))
    print(f"\n[json] {out}")


if __name__ == "__main__":
    main()
