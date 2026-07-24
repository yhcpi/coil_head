"""v48: v39 + RANSAC pool cached on GPU.

Replaces per-round `skel_pixels[~claimed_mask]` construction +
fresh GPU upload + new distance matrix. Instead:
  - Upload full pool ONCE
  - Per round: build `free_mask` on CPU, index pool_t[free_mask] on GPU
  - Reuse pool_t for distance matrix

Expected saving: 5-15ms total (small pool upload was repeated).
"""
from __future__ import annotations
import math
import time
from typing import Any

import cv2
import numpy as np
import torch

import v32
import v35
import v38

PALETTE = v32.PALETTE


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cached_ransac(pool_t_full, n_pixels, *, n_iter, rng,
                   r_min, r_max, band_px, min_support,
                   claimed_t, dev, r_pool, H, W):
    """One RANSAC round using pre-uploaded GPU pool.

    pool_t_full: (N, 2) tensor on GPU
    claimed_t: (N,) bool tensor on GPU (latest claimed mask)
    Returns (cx, cy, r, score, sigma, free_pixel_indices_cpu) or None
    """
    free_mask = ~claimed_t
    if int(free_mask.sum().item()) < 30:
        return None
    free_idx_local = torch.where(free_mask)[0]  # indices into pool_t_full
    n_free = len(free_idx_local)
    pool_t = pool_t_full[free_idx_local]  # (n_free, 2)
    n_total = n_iter
    replace = (n_total > n_free)
    rng_np = np.random.default_rng(rng.integers(0, 2**32))
    sel = rng_np.choice(n_free, size=(n_total, 3), replace=replace or True)
    sel_t = torch.from_numpy(sel).to(dev)
    p1 = pool_t[sel_t[:, 0]]
    p2 = pool_t[sel_t[:, 1]]
    p3 = pool_t[sel_t[:, 2]]
    cxs, cys, rs, valid = v35._circle_from_3pts_batch_gpu(
        p1, p2, p3, r_min=r_min, r_max=r_max)
    if not valid.any():
        return None
    valid_idx = torch.where(valid)[0]
    v_cx = cxs[valid]; v_cy = cys[valid]; v_r = rs[valid]
    dx = pool_t[None, :, 0] - v_cx[:, None]
    dy = pool_t[None, :, 1] - v_cy[:, None]
    dists = torch.sqrt(dx * dx + dy * dy)
    residuals = (dists - v_r[:, None]).abs()
    inliers_init = residuals < band_px
    scores = inliers_init.sum(dim=1)
    best = int(scores.argmax().item())
    score = int(scores[best].item())
    if score < min_support:
        return None
    cx = float(v_cx[best].item())
    cy = float(v_cy[best].item())
    r0 = float(v_r[best].item())
    inlier_mask_local = inliers_init[best]
    inlier_global_idx = free_idx_local[inlier_mask_local].cpu().numpy()
    inlier_res = residuals[best][inlier_mask_local]
    if len(inlier_res) >= 5:
        med = float(inlier_res.median().item())
        mad = float((inlier_res - med).abs().median().item())
        sigma = 1.4826 * mad
    else:
        sigma = 5.0
    return cx, cy, r0, score, sigma, inlier_global_idx, pool_t


def fit_v48(mask_bin, *,
            n_rounds=12, n_iter=200,
            band_px=5.0, min_support=80,
            r_min_factor=0.10, r_max_factor=0.95,
            ratio_min=1.00, ratio_max=1.10,
            support_min=400, dr_tol_px=5.0,
            k_sigma=3.0, sigma_max=8.0,
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
    dev = _device()
    rng = np.random.default_rng(rng_seed)
    pool_t_full = torch.from_numpy(skel_pixels.astype(np.int32)).to(dev)
    n_pix = len(skel_pixels)
    claimed_t = torch.zeros(n_pix, dtype=torch.bool, device=dev)
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
    for round_idx in range(n_rounds):
        res = _cached_ransac(pool_t_full, n_pix, n_iter=n_iter, rng=rng,
                             r_min=r_min_est, r_max=r_max, band_px=band_px,
                             min_support=min_support, claimed_t=claimed_t,
                             dev=dev, r_pool=r_max, H=H, W=W)
        if res is None:
            break
        cx0, cy0, r0, score, sigma, inlier_global_idx, pool_t = res
        # Update claimed mask: inliers
        claimed_t[torch.from_numpy(inlier_global_idx).to(dev)] = True
        if sigma > sigma_max:
            continue
        # Re-select with k*sigma — still on GPU
        free_idx_local = torch.where(~claimed_t)[0]
        # Need inlier_pix inlier set to find new_mask — but inlier_global_idx
        # already excludes claimed ones. Use those:
        inlier_pix = skel_pixels[inlier_global_idx]
        dx0 = skel_pixels[free_idx_local.cpu().numpy()][:, 0] - cx0
        dy0 = skel_pixels[free_idx_local.cpu().numpy()][:, 1] - cy0
        d0 = np.sqrt(dx0 * dx0 + dy0 * dy0)
        new_mask = np.abs(d0 - r0) < (k_sigma * sigma)
        if new_mask.sum() < min_support:
            continue
        ref = v32._fit_ellipse_safe(
            skel_pixels[free_idx_local.cpu().numpy()][new_mask].reshape(-1, 1, 2))
        if ref is None:
            continue
        cx, cy, a, b, ang = ref
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
            support_per_r, low_sr=arc_lo_sr, high_sr=arc_hi_sr,
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
            "src": "v48_cached_pool",
        })
        # Reclaim band in claimed_t (CPU computations -> upload)
        d_full = np.sqrt((skel_pixels[:, 0] - cx) ** 2 +
                          (skel_pixels[:, 1] - cy) ** 2)
        in_band = np.abs(d_full - r_ell) <= band_px
        claimed_t[torch.from_numpy(np.where(in_band)[0]).to(dev)] = True
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
        "elapsed_s": {
            "skel": t1 - t0, "ransac": t2 - t1,
            "dedup": t3 - t2, "total": t3 - t0,
        },
    }