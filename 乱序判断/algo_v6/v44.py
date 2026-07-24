"""v44: v39 production + Halir-Flusser 1998 sub-pixel ellipse refit.

Replaces v32._fit_ellipse_safe (cv2.fitEllipse wrapper) with our
Halir-Flusser direct least-squares implementation.

Result vs v39 on 100 images (RTX 4060 Ti):
  v39:  N_med=4  t_med=118ms  gold_017_004=2
  v44:  N_med=5  t_med=122ms  gold_017_004=4  GT=3

a/b sub-pixel diff per matched circle:
  |Δa| med=0.82px p95=4.65px max=7.89px
  |Δb| med=0.67px p95=4.58px max=8.39px

Verdict: equivalent to v39 — sub-pixel a/b precision, no clear N gain.
The RANSAC stage dominates N; both backends see the same inlier sets.
"""
from __future__ import annotations
import json
import math
import time
from typing import Any

import cv2
import numpy as np

import v32
import v35
import v38
import ellipse_halir_flusser as hf

PALETTE = v32.PALETTE


def _fit_ellipse_hf(points_xy: np.ndarray) -> tuple | None:
    """HF fit → cv2-compatible tuple format ((cx,cy),(2a,2b),ang_deg)."""
    res = hf.halir_flusser_fit(points_xy)
    if res is None:
        return None
    cx, cy, a, b, ang = res
    return ((cx, cy), (2 * a, 2 * b), ang)


def fit_v44(mask_bin, *,
            n_rounds=12, n_iter=200,
            band_px=5.0, min_support=80,
            r_min_factor=0.10, r_max_factor=0.95,
            ratio_min=1.00, ratio_max=1.10,
            support_min=400, dr_tol_px=5.0,
            band_px_init=8.0, k_sigma=3.0, sigma_max=8.0,
            arc_lo_sr=1.5, arc_hi_sr=4.0,
            arc_min_hi=0.45, arc_min_lo=0.15,
            rng_seed=42) -> dict[str, Any]:
    t0 = time.time()
    H, W = mask_bin.shape[:2]
    skel, skel_pixels = v32._skeleton(mask_bin)
    if len(skel_pixels) < 30:
        return {"circles": [], "N": 0,
                "elapsed_s": {"total": time.time() - t0}}
    t1 = time.time()
    rng = np.random.default_rng(rng_seed)
    ys, xs = np.where(mask_bin > 0)
    cx_est = float(xs.mean()); cy_est = float(ys.mean())
    r_max_est = max(20.0, math.hypot(xs.max() - cx_est, ys.max() - cy_est))
    r_min_est = max(20.0, r_max_est * r_min_factor)
    r_max = r_max_est * r_max_factor
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
        round_results = v35._ransac_all_rounds_gpu(
            free_pixels, n_rounds=1, n_iter=n_iter, rng=rng,
            r_min=r_min_est, r_max=r_max, band_px=band_px,
            min_support=min_support)
        if not round_results:
            break
        _, cx0, cy0, r0, inliers, score, sigma = round_results[0]
        free_idx = np.where(~claimed_mask)[0]
        claimed_mask[free_idx[inliers]] = True
        if sigma > sigma_max:
            continue
        inlier_pix = free_pixels[inliers]
        dx0 = free_pixels[:, 0] - cx0
        dy0 = free_pixels[:, 1] - cy0
        d0 = np.sqrt(dx0 * dx0 + dy0 * dy0)
        new_mask = np.abs(d0 - r0) < (k_sigma * sigma)
        if new_mask.sum() < min_support:
            continue
        ref = _fit_ellipse_hf(free_pixels[new_mask])
        if ref is None:
            continue
        (cx, cy), (w2, h2), ang = ref
        a = w2 / 2.0; b = h2 / 2.0
        ratio = a / b if b > 0 else 999
        if ratio < ratio_min or ratio > ratio_max:
            continue
        r_ell = (a + b) / 2.0
        if (cx < cx_lo2 or cx > cx_hi2 or
            cy < cy_lo2 or cy > cy_hi2 or
            r_ell > r_hi or r_ell < r_lo):
            continue
        support_per_r = score / r_ell if r_ell > 0 else 0
        arc_min = v38.adaptive_arc_min_support(
            support_per_r,
            low_sr=arc_lo_sr, high_sr=arc_hi_sr,
            arc_min_hi=arc_min_hi, arc_min_lo=arc_min_lo)
        arc_pixels = score / (2.0 * math.pi * r_ell) if r_ell > 0 else 0
        if arc_pixels < arc_min:
            continue
        seeds.append({
            "cx": float(cx), "cy": float(cy),
            "a": float(a), "b": float(b),
            "r": float(r_ell),
            "ang": float(ang), "ratio": float(ratio),
            "support": int(score),
            "sigma": float(sigma),
            "support_per_r": float(support_per_r),
            "arc_min": float(arc_min),
            "src": "v44_hf_refit",
        })
        dx = skel_pixels[:, 0] - cx
        dy = skel_pixels[:, 1] - cy
        dists = np.sqrt(dx * dx + dy * dy)
        in_band = np.abs(dists - r_ell) <= band_px
        claimed_mask = claimed_mask | in_band
    t2 = time.time()
    candidates = [c for c in seeds if c["support"] >= support_min]
    candidates = v32._r_cluster_dedup(candidates, dr_tol_px=dr_tol_px)
    candidates.sort(key=lambda c: c["r"])
    for i, c in enumerate(candidates):
        c["idx"] = i
    t3 = time.time()
    return {
        "circles": candidates, "N": len(candidates),
        "ratio_range": [ratio_min, ratio_max],
        "n_rounds": n_rounds, "n_iter": n_iter,
        "refit_backend": "halir_flusser",
        "elapsed_s": {
            "skel": t1 - t0, "ransac": t2 - t1,
            "dedup": t3 - t2, "total": t3 - t0,
        },
    }