"""Sweep n_rounds × n_iter to find best speed/efficiency trade-off."""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Patch v35.fit_v35 to accept n_rounds/n_iter params
import v35 as v35_mod

here = Path(__file__).resolve().parent
project = here.parent
MASKS_DIR = project / "yhc" / "mask_refine"


def run_with_params(mask_bin, n_rounds, n_iter):
    """Replicate v35.fit_v35 logic but with configurable n_rounds/n_iter."""
    import math
    import numpy as np
    import v32
    t0 = time.time()
    H, W = mask_bin.shape[:2]
    skel, skel_pixels = v32._skeleton(mask_bin)
    if len(skel_pixels) < 30:
        return {"N": 0, "t_ms": (time.time()-t0)*1000}
    t1 = time.time()
    rng = np.random.default_rng(42)
    ys, xs = np.where(mask_bin > 0)
    cx_est = float(xs.mean()); cy_est = float(ys.mean())
    r_max_est = max(20.0, math.hypot(xs.max() - cx_est, ys.max() - cy_est))
    r_min_est = max(20.0, r_max_est * 0.10)
    r_max = r_max_est * 0.95
    cx_lo2, cx_hi2 = -W * 0.05, W * 1.05
    cy_lo2, cy_hi2 = -H * 0.05, H * 1.05
    r_lo = max(80.0, r_max_est * 0.30)
    r_hi = min(max(W, H) * 0.50, 700.0)
    seeds = []
    claimed_mask = np.zeros(len(skel_pixels), dtype=bool)
    for round_idx in range(n_rounds):
        free_pixels = skel_pixels[~claimed_mask]
        if len(free_pixels) < 30:
            break
        round_results = v35_mod._ransac_all_rounds_gpu(
            free_pixels, n_rounds=1, n_iter=n_iter, rng=rng,
            r_min=r_min_est, r_max=r_max, band_px=5.0,
            min_support=80)
        if not round_results:
            break
        _, cx0, cy0, r0, inliers, score, sigma = round_results[0]
        free_idx = np.where(~claimed_mask)[0]
        claimed_mask[free_idx[inliers]] = True
        if sigma > 8.0:
            continue
        inlier_pix = free_pixels[inliers]
        dx0 = free_pixels[:, 0] - cx0
        dy0 = free_pixels[:, 1] - cy0
        d0 = np.sqrt(dx0 * dx0 + dy0 * dy0)
        new_mask = np.abs(d0 - r0) < (3.0 * sigma)
        if new_mask.sum() < 80:
            continue
        ref = v32._fit_ellipse_safe(
            free_pixels[new_mask].reshape(-1, 1, 2))
        if ref is None:
            continue
        cx, cy, a, b, ang = ref
        ratio = a / b if b > 0 else 999
        if ratio < 1.00 or ratio > 1.10:
            continue
        r_ell = (a + b) / 2.0
        if (cx < cx_lo2 or cx > cx_hi2 or
            cy < cy_lo2 or cy > cy_hi2 or
            r_ell > r_hi or r_ell < r_lo):
            continue
        arc_pixels = score / (2.0 * math.pi * r_ell) if r_ell > 0 else 0
        if arc_pixels < 0.20:
            continue
        seeds.append({
            "cx": float(cx), "cy": float(cy),
            "a": float(a), "b": float(b),
            "r": float(r_ell),
            "support": int(score),
        })
    candidates = [c for c in seeds if c["support"] >= 400]
    candidates = v32._r_cluster_dedup(candidates, dr_tol_px=5.0)
    candidates.sort(key=lambda c: c["r"])
    return {"N": len(candidates),
            "t_ms": (time.time()-t0)*1000,
            "t_ransac_ms": (time.time()-t1)*1000}


def main():
    files = sorted(MASKS_DIR.glob("*.png"))
    # Use 30 random masks for fast sweep
    np.random.seed(0)
    sample_files = list(np.random.choice(files, 30, replace=False))
    # Warm up
    img0 = cv2.imread(str(sample_files[0]), cv2.IMREAD_GRAYSCALE)
    m0 = (img0 > 127).astype(np.uint8)
    run_with_params(m0, 6, 200)
    torch.cuda.synchronize()

    # Configurations to sweep
    configs = [
        (12, 300),  # current v35
        (12, 200),
        (12, 150),
        (8, 200),
        (8, 150),
        (6, 300),
        (6, 200),
        (6, 150),
        (4, 200),
        (4, 300),
    ]
    print(f"Sweeping {len(configs)} configs on {len(sample_files)} images")
    results = {}
    for n_r, n_i in configs:
        ns = []
        ts = []
        for f in sample_files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            mask = (img > 127).astype(np.uint8)
            r = run_with_params(mask, n_r, n_i)
            ns.append(r["N"])
            ts.append(r["t_ms"])
            torch.cuda.synchronize()
        results[(n_r, n_i)] = {
            "n_med": float(np.median(ns)),
            "n_mean": float(np.mean(ns)),
            "t_med": float(np.median(ts)),
            "t_p95": float(np.percentile(ts, 95)),
        }
        print(f"  rounds={n_r:>2} iter={n_i:>3}  "
              f"N_med={results[(n_r, n_i)]['n_med']:.0f}  "
              f"N_mean={results[(n_r, n_i)]['n_mean']:.1f}  "
              f"t_med={results[(n_r, n_i)]['t_med']:.0f}ms  "
              f"t_p95={results[(n_r, n_i)]['t_p95']:.0f}ms")
    # Best config: maximum N_mean per ms
    print("\nEfficiency (N_mean / t_med):")
    eff = [(k, v["n_mean"] / max(1, v["t_med"]) * 1000)
           for k, v in results.items()]
    eff.sort(key=lambda x: -x[1])
    for k, e in eff:
        v = results[k]
        print(f"  {k}: {e:.2f} wires/s (N_med={v['n_med']:.0f}, t_med={v['t_med']:.0f}ms)")

    import json
    out = project / "algo_v6" / "v37_sweep.json"
    out.write_text(json.dumps(
        {f"{k[0]}x{k[1]}": v for k, v in results.items()},
        indent=2))
    print(f"\n[json] {out}")


if __name__ == "__main__":
    main()