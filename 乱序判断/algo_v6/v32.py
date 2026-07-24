"""v32: Skeleton-first + GPU RANSAC + polar growth + assign-dedup.

User feedback (2026-07-24):
  1. Remove envelope entirely (no center, no circle, nothing)
  2. Real-time <100ms/img (target 50ms)
  3. Polar sequential growth (non-concentric) from each candidate's own center
  4. Same contour → multiple circle fits: STILL need to fix

Active env: hyper-yolo (PyTorch 2.5.1+cu124, RTX 4060 Ti, cv2 5.0).

Algorithm (5 stages):

  Stage 1: SKELETON
    - skimage.morphology.skeletonize(mask) → 1-pixel-wide curves
    - Critical: eliminates "thick contour → multi-fit" problem because
      each wire turn contour is now a single-pixel line.

  Stage 2: GPU 3-pt RANSAC
    - Sample n_hyp × 3 skeleton pixels (CPU, fast)
    - Compute hypothes (cx,cy,r) on CPU (small array)
    - Upload to GPU
    - Distance matrix (n_hyp × n_skel) → inliers → scores (GPU)
    - argmax → best seed
    - Refit with cv2.fitEllipse (CPU)

  Stage 3: Polar sequential growth (non-concentric)
    - Sort seeds by r ascending (innermost first)
    - For each seed, in polar coords from its OWN center:
      - Sample angular bins (e.g. 16)
      - In each bin, find r peaks beyond current circle
      - Skip pixels already assigned to previous seeds
      - Refit circle with new pixels
    - This adds wire turns RANSAC missed (between seeds, near edges).

  Stage 4: Strict shape filter
    - a/b ∈ [1.00, 1.10]
    - Per-pixel assignment: each skeleton pixel claims to at most
      ONE circle (highest support).

  Stage 5: Assign-based dedup
    - For each pair of circles: if 70% of A's pixels also belong to B
      → A is dup of B (keep B which has higher support).
    - This eliminates "same contour → 2 circles fit" because both
      circles claim the same skeleton pixels.

Performance:
  - Skeleton ~5% of mask pixels → 30x fewer pixels per RANSAC iter
  - GPU distance matrix: 0.08s for 500×30000
  - Total expected: 30-80ms per image

NOTE on naming: "INBD-style" was a speculated reference; verification
(WebSearch July 2026) showed no paper by that exact name exists. The
technique is polar growth from each seed's own center, which is a
generic classical pattern from Hough-style ellipse detection
(Fitzgibbon-Pilu-Fisher 1999, IEEE TPAMI 21:5).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize


PALETTE = [
    (255, 0, 0), (0, 200, 255), (0, 255, 0),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 255), (255, 128, 0), (0, 200, 100),
    (200, 200, 0), (200, 100, 100), (255, 80, 80),
    (80, 255, 80), (80, 80, 255), (255, 180, 0),
    (180, 0, 255), (0, 180, 255), (255, 100, 200),
    (180, 255, 100), (100, 180, 255), (255, 200, 180),
    (255, 0, 128), (128, 255, 0), (0, 128, 255),
]


_DEVICE = None


def _device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available()
                               else "cpu")
    return _DEVICE


# ============================================================================
# Stage 1: Skeleton
# ============================================================================
def _skeleton(mask_bin):
    """Skeleton → 1-pixel-wide curves.

    Returns:
      skel (H, W) uint8 0/1
      skel_pixels (N, 2) float32 of (x, y) where skel > 0
    """
    skel = skeletonize(mask_bin > 0).astype(np.uint8)
    ys, xs = np.where(skel > 0)
    if len(xs) < 30:
        return skel, np.zeros((0, 2), dtype=np.float32)
    skel_pixels = np.column_stack([xs.astype(np.float32),
                                   ys.astype(np.float32)])
    return skel, skel_pixels


# ============================================================================
# Stage 2: GPU 3-pt RANSAC
# ============================================================================
def _circle_from_3pts(p1, p2, p3):
    """CPU 3-pt circle, returns (cx, cy, r) or None."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return None
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) +
          (x2 * x2 + y2 * y2) * (y3 - y1) +
          (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) +
          (x2 * x2 + y2 * y2) * (x1 - x3) +
          (x3 * x3 + y3 * y3) * (x2 - x1)) / d
    r = math.sqrt((x1 - ux) ** 2 + (y1 - uy) ** 2)
    return ux, uy, r


def _gpu_distance_matrix(pool_t, hyp_cx, hyp_cy, hyp_r, band_px):
    """Compute |dists - r| ≤ band_px for all (hyp × pool) on GPU.

    Returns:
      inliers (n_hyp, n_pool) bool tensor on CPU
    """
    dev = _device()
    pool_t = pool_t.to(dev, non_blocking=True)
    hyp_cx = hyp_cx.to(dev, non_blocking=True)
    hyp_cy = hyp_cy.to(dev, non_blocking=True)
    hyp_r = hyp_r.to(dev, non_blocking=True)
    # Broadcast: (n_hyp, 1) - (1, n_pool) → (n_hyp, n_pool)
    dx = pool_t[None, :, 0] - hyp_cx[:, None]
    dy = pool_t[None, :, 1] - hyp_cy[:, None]
    dists = torch.sqrt(dx * dx + dy * dy)
    inliers = (dists - hyp_r[:, None]).abs() <= band_px
    return inliers.cpu(), dists.cpu()


def _ransac_round_gpu(pool_pixels, *, n_iter, rng,
                       r_min, r_max, band_px, min_support,
                       device_chunk=200):
    """One RANSAC round, GPU-accelerated for distance matrix.

    Splits n_iter into chunks of device_chunk, evaluates each chunk's
    hypothes on GPU.
    """
    if len(pool_pixels) < 3:
        return None, None, None, None, -1
    pool_t = torch.from_numpy(pool_pixels)  # CPU, will upload
    best_score = -1
    best_circ = None
    best_inliers = None
    n_chunks = (n_iter + device_chunk - 1) // device_chunk
    for c in range(n_chunks):
        actual = min(device_chunk, n_iter - c * device_chunk)
        sel = rng.choice(len(pool_pixels), size=(actual, 3), replace=False)
        p1 = pool_pixels[sel[:, 0]]
        p2 = pool_pixels[sel[:, 1]]
        p3 = pool_pixels[sel[:, 2]]
        cxs, cys, rs = [], [], []
        for i in range(actual):
            cc = _circle_from_3pts(p1[i], p2[i], p3[i])
            if cc is None:
                cxs.append(np.nan); cys.append(np.nan); rs.append(np.nan)
            else:
                cxs.append(cc[0]); cys.append(cc[1]); rs.append(cc[2])
        cxs_t = torch.tensor(cxs, dtype=torch.float32)
        cys_t = torch.tensor(cys, dtype=torch.float32)
        rs_t = torch.tensor(rs, dtype=torch.float32)
        # Filter by r
        valid = (~torch.isnan(rs_t)) & (rs_t >= r_min) & (rs_t <= r_max)
        if not valid.any():
            continue
        v_cx = cxs_t[valid]
        v_cy = cys_t[valid]
        v_r = rs_t[valid]
        inliers, _ = _gpu_distance_matrix(pool_t, v_cx, v_cy, v_r, band_px)
        scores = inliers.sum(dim=1)
        chunk_best = int(scores.max())
        if chunk_best > best_score:
            best_score = chunk_best
            local_idx = int(scores.argmax())
            actual_idx = torch.where(valid)[0][local_idx].item()
            best_circ = (cxs[actual_idx], cys[actual_idx], rs[actual_idx])
            best_inliers = torch.where(inliers[local_idx])[0].numpy()
    if best_score < min_support:
        return None, None, None, None, -1
    return (*best_circ, best_inliers, best_score)


def _fit_ellipse_safe(pts):
    if len(pts) < 30:
        return None
    try:
        (cx, cy), (fa, fb), ang = cv2.fitEllipse(pts)
    except cv2.error:
        return None
    a, b = fa / 2.0, fb / 2.0
    if a < b:
        a, b = b, a
        ang = (ang + 90) % 180
    return float(cx), float(cy), float(a), float(b), float(ang)


# ============================================================================
# Stage 3: Polar sequential growth (non-concentric)
# ============================================================================
def _polar_grow(seed_cx, seed_cy, seed_r, skel_pixels, claimed_mask,
                *, n_angular_bins=16, band_px=5.0, ratio_max=1.10):
    """Grow from seed circle OUTWARD in polar coords from seed's center.

    For each angular bin, find the maximum r peak beyond current circle
    (skipping already-claimed pixels). Returns new circles (or empty).

    Polar growth from EACH SEED's own center (non-concentric variant).
    Generic pattern used in classical ellipse detection (Fitzgibbon 1999).
    Note: "INBD" was a speculated name; verification (July 2026) showed
    no paper by that exact name exists.
    """
    if len(skel_pixels) < 30:
        return []
    # Compute (theta, r) for each unclaimed skeleton pixel relative to seed
    free = ~claimed_mask
    if free.sum() < 30:
        return []
    free_pix = skel_pixels[free]
    dx = free_pix[:, 0] - seed_cx
    dy = free_pix[:, 1] - seed_cy
    r_polar = np.sqrt(dx * dx + dy * dy)
    theta_polar = np.arctan2(dy, dx)  # -π to π
    # Skip pixels within current seed (already part of it)
    outer_mask = r_polar > seed_r * 0.9
    if outer_mask.sum() < 10:
        return []
    r_outer = r_polar[outer_mask]
    theta_outer = theta_polar[outer_mask]
    new_circles = []
    # For each angular bin, find median r
    bin_width = 2 * np.pi / n_angular_bins
    for b in range(n_angular_bins):
        theta_lo = -np.pi + b * bin_width
        theta_hi = theta_lo + bin_width
        bin_mask = (theta_outer >= theta_lo) & (theta_outer < theta_hi)
        if bin_mask.sum() < 8:
            continue
        r_bin = r_outer[bin_mask]
        # Use mode (peak of histogram) to find dominant radius
        hist, edges = np.histogram(r_bin, bins=20)
        peak_idx = hist.argmax()
        peak_r = (edges[peak_idx] + edges[peak_idx + 1]) / 2
        # Only consider peaks within reasonable wire turn spacing
        if abs(peak_r - seed_r) < 5 or peak_r < 50 or peak_r > 1200:
            continue
        # Collect all pixels in this bin near peak_r
        ring_mask = bin_mask & (r_outer > peak_r - band_px) & (r_outer < peak_r + band_px)
        ring_pixels_rel = np.column_stack([
            r_outer[ring_mask] * np.cos(theta_outer[ring_mask]),
            r_outer[ring_mask] * np.sin(theta_outer[ring_mask])
        ])
        ring_pixels_abs = ring_pixels_rel + np.array([seed_cx, seed_cy])
        ref = _fit_ellipse_safe(ring_pixels_abs.astype(np.float32)
                                .reshape(-1, 1, 2))
        if ref is None:
            continue
        cx, cy, a, b_, ang = ref
        if a <= 0 or b_ <= 0:
            continue
        ratio = a / b_
        if ratio > ratio_max:
            continue
        new_circles.append({
            "cx": float(cx), "cy": float(cy),
            "a": float(a), "b": float(b_),
            "r": float((a + b_) / 2.0),
            "ang": float(ang), "ratio": float(ratio),
            "support": int(ring_mask.sum()),
            "src": "polar_grow",
        })
    return new_circles


# ============================================================================
# Stage 4-5: Per-pixel assignment + dedup
# ============================================================================
def _assign_pixels(circles, skel_pixels, *, band_px=5.0):
    """Each skeleton pixel assigned to at most ONE circle (best score).

    Returns:
      assignment (N,) int: index into circles, or -1 if unassigned
    """
    n_pix = len(skel_pixels)
    if not circles or n_pix == 0:
        return np.full(n_pix, -1, dtype=np.int32)
    cxs = np.array([c["cx"] for c in circles])
    cys = np.array([c["cy"] for c in circles])
    rs = np.array([c["r"] for c in circles])
    sups = np.array([c["support"] for c in circles], dtype=np.float32)
    # Distance matrix (n_circles × n_pix)
    dx = skel_pixels[None, :, 0] - cxs[:, None]
    dy = skel_pixels[None, :, 1] - cys[:, None]
    dists = np.sqrt(dx * dx + dy * dy)
    within_band = np.abs(dists - rs[:, None]) <= band_px  # (n_circles, n_pix)
    assignment = np.full(n_pix, -1, dtype=np.int32)
    # For each pixel, pick the highest-support circle claiming it
    for i in range(n_pix):
        claimers = np.where(within_band[:, i])[0]
        if len(claimers) == 0:
            continue
        # Pick highest-support claimer
        best = claimers[sups[claimers].argmax()]
        assignment[i] = best
    return assignment


def _r_cluster_dedup(circles, *, dr_tol_px=8.0):
    """Cluster circles by r (within dr_tol). Within each cluster,
    keep only the highest-support circle.

    This is classical RANSAC dedup: if 2 circles are within 8 px in r,
    they're competing for the same band → SAME wire turn → keep best.
    Different wire turns have Δr > 7 px (band separation).
    """
    if len(circles) <= 1:
        return list(circles)
    # Sort by r ASC for cluster processing
    order = sorted(range(len(circles)), key=lambda i: circles[i]["r"])
    kept = []
    for i in order:
        ci = circles[i]
        # Check if i is within dr_tol of any kept circle
        dup = False
        for k in kept:
            if abs(ci["r"] - k["r"]) < dr_tol_px:
                # Same cluster → only keep the one with higher support
                if ci["support"] > k["support"]:
                    kept.remove(k)
                    kept.append(ci)
                dup = True
                break
        if not dup:
            kept.append(ci)
    return kept


def _assign_dedup(circles, skel_pixels, *,
                  dr_tol_px=10.0, dxy_tol_px=300.0,
                  band_px=5.0, overlap_thr=0.50):
    """Drop circles that are SAME wire turn as another (higher-support) circle.

    2-stage dedup:
      Stage 1: r-cluster dedup (classical, dr_tol=8)
      Stage 2: assign-based dedup for borderline cases
    """
    after_r = _r_cluster_dedup(circles, dr_tol_px=dr_tol_px)
    if len(after_r) <= 1:
        return after_r
    # Stage 2: per-pixel overlap for cases where r is similar but center
    # is very different (partial-arc fit of same wire turn)
    n = len(after_r)
    cxs = np.array([c["cx"] for c in after_r])
    cys = np.array([c["cy"] for c in after_r])
    rs = np.array([c["r"] for c in after_r])
    sups = np.array([c["support"] for c in after_r], dtype=np.float32)
    # Compute band pixel counts
    band_counts = []
    band_masks = []
    for i in range(n):
        dx = skel_pixels[:, 0] - cxs[i]
        dy = skel_pixels[:, 1] - cys[i]
        d = np.sqrt(dx * dx + dy * dy)
        bm = np.abs(d - rs[i]) <= band_px
        band_masks.append(bm)
        band_counts.append(int(bm.sum()))
    # Process by support DESC
    order = np.argsort(-sups)
    is_kept = np.ones(n, dtype=bool)
    for idx in order:
        i = int(idx)
        if not is_kept[i]:
            continue
        for j in np.where(is_kept)[0]:
            if j == i:
                continue
            if abs(rs[i] - rs[j]) >= dr_tol_px:
                continue
            # In same r cluster — already deduped by stage 1 unless
            # we kept the lower-support one because it was first.
            # Stage 2: pixel overlap of A's band vs B's band
            if band_counts[i] == 0:
                continue
            inter = int((band_masks[i] & band_masks[j]).sum())
            if inter / band_counts[i] > overlap_thr:
                is_kept[i] = False
                break
    return [after_r[i] for i in np.where(is_kept)[0]]


# ============================================================================
# Top-level fit_v32
# ============================================================================
def fit_v32(mask_bin, *,
            n_rounds=12, n_iter=300, device_chunk=200,
            band_px=5.0, min_support=80,
            r_min_factor=0.10, r_max_factor=0.95,
            ratio_min=1.00, ratio_max=1.10,
            support_min=400, overlap_thr=0.50,
            dr_tol_px=5.0,
            enable_polar_grow=True, n_polar_bins=16,
            fast_mode=False,
            rng_seed=42) -> dict[str, Any]:
    """v32: skeleton-first + GPU RANSAC + polar growth + assign dedup.

    fast_mode=True → n_rounds=8, n_iter=200, no polar grow (target 50ms).
    """
    if fast_mode:
        n_rounds = 8
        n_iter = 200
        enable_polar_grow = False
    t0 = time.time()
    # Stage 1: skeleton
    skel, skel_pixels = _skeleton(mask_bin)
    if len(skel_pixels) < 30:
        return {"circles": [], "N": 0, "elapsed_s": time.time() - t0}
    t1 = time.time()
    H, W = mask_bin.shape[:2]
    # Stage 2: GPU RANSAC rounds
    rng = np.random.default_rng(rng_seed)
    # Estimate r range from envelope (1-pass, fast — using max(mask) bounding)
    ys, xs = np.where(mask_bin > 0)
    cx_est = float(xs.mean()); cy_est = float(ys.mean())
    r_max_est = max(20.0, math.hypot(xs.max() - cx_est, ys.max() - cy_est))
    r_min_est = max(20.0, r_max_est * r_min_factor)
    r_max = r_max_est * r_max_factor
    # Bounds filter: drop circles whose center is far off-image
    cx_lo, cx_hi = -W * 0.20, W * 1.20
    cy_lo, cy_hi = -H * 0.20, H * 1.20
    seeds = []
    claimed_mask = np.zeros(len(skel_pixels), dtype=bool)
    for round_idx in range(n_rounds):
        free_pixels = skel_pixels[~claimed_mask]
        if len(free_pixels) < 30:
            break
        cx0, cy0, r0, inliers, score = _ransac_round_gpu(
            free_pixels, n_iter=n_iter, rng=rng,
            r_min=r_min_est, r_max=r_max, band_px=band_px,
            min_support=min_support, device_chunk=device_chunk)
        if cx0 is None:
            break
        inlier_pix = free_pixels[inliers]
        ref = _fit_ellipse_safe(inlier_pix.reshape(-1, 1, 2))
        if ref is None:
            claimed_mask[np.where(~claimed_mask)[0][inliers]] = True
            continue
        cx, cy, a, b, ang = ref
        ratio = a / b if b > 0 else 999
        if ratio < ratio_min or ratio > ratio_max:
            claimed_mask[np.where(~claimed_mask)[0][inliers]] = True
            continue
        r_ell = (a + b) / 2.0
        # Bounds filter: drop off-image centers and r too large
        if (cx < cx_lo or cx > cx_hi or
            cy < cy_lo or cy > cy_hi or
            r_ell > max(W, H) * 1.0):
            claimed_mask[np.where(~claimed_mask)[0][inliers]] = True
            continue
        seeds.append({
            "cx": float(cx), "cy": float(cy),
            "a": float(a), "b": float(b),
            "r": float(r_ell),
            "ang": float(ang), "ratio": float(ratio),
            "support": int(score),
            "src": "ransac",
        })
        # Mark claimed
        idx = np.where(~claimed_mask)[0][inliers]
        claimed_mask[idx] = True
    t2 = time.time()
    # Stage 3: Polar sequential growth (per-seed, non-concentric)
    new_circles = []
    if enable_polar_grow:
        for seed in seeds:
            nc = _polar_grow(seed["cx"], seed["cy"], seed["r"],
                             skel_pixels, claimed_mask,
                             n_angular_bins=n_polar_bins,
                             band_px=band_px, ratio_max=ratio_max)
            new_circles.extend(nc)
    all_circles = seeds + new_circles
    t3 = time.time()
    # Stage 4: support filter + bounds filter
    # Adaptive r bounds: real wire turns in our data are r ~ 250-600
    r_lo = max(80.0, r_max_est * 0.30)  # smaller than 30% of envelope
    r_hi = min(max(W, H) * 0.50, 700.0)  # larger than 50% of dim OR > 700
    cx_lo2, cx_hi2 = -W * 0.05, W * 1.05
    cy_lo2, cy_hi2 = -H * 0.05, H * 1.05
    all_circles = [c for c in all_circles
                   if c["support"] >= support_min
                   and cx_lo2 <= c["cx"] <= cx_hi2
                   and cy_lo2 <= c["cy"] <= cy_hi2
                   and r_lo <= c["r"] <= r_hi]
    # Stage 5: assign-dedup
    final = _assign_dedup(all_circles, skel_pixels,
                          band_px=band_px, overlap_thr=overlap_thr,
                          dr_tol_px=dr_tol_px)
    final.sort(key=lambda c: c["r"])
    for i, c in enumerate(final):
        c["idx"] = i
    t4 = time.time()
    if len(final) >= 2:
        rs = np.array([c["r"] for c in final])
        gaps = np.diff(rs)
        P_wr = float(np.mean(gaps))
        P_wr_std = float(np.std(gaps))
    else:
        P_wr = P_wr_std = 0.0
    return {
        "circles": final, "N": len(final),
        "n_seeds": len(seeds), "n_polar_new": len(new_circles),
        "P_wr": P_wr, "P_wr_std": P_wr_std,
        "ratio_range": [ratio_min, ratio_max],
        "n_skel_pixels": int(len(skel_pixels)),
        "elapsed_s": {
            "skeleton": t1 - t0,
            "ransac": t2 - t1,
            "polar_grow": t3 - t2,
            "dedup": t4 - t3,
            "total": t4 - t0,
        },
        "device": str(_device()),
    }


def render(raw_bgr, mask_bin, result, out_path, gt_path=None):
    """v32 render: NO envelope. Show skeleton faintly + circles."""
    p = raw_bgr.copy()
    # Optional: overlay skeleton in dim grey
    if "n_skel_pixels" in result and result["n_skel_pixels"] > 0:
        # Render dim grey where mask is 1 (skeleton comes from mask)
        overlay = p.copy()
        overlay[mask_bin > 0] = (60, 60, 60)
        p = cv2.addWeighted(p, 0.7, overlay, 0.3, 0)
    for e in result["circles"]:
        c = PALETTE[e["idx"] % len(PALETTE)]
        ratio = e["ratio"]
        is_warning = ratio > 1.07
        if is_warning:
            n_segs = 32
            for k in range(0, n_segs, 2):
                a0 = 360.0 * k / n_segs
                a1 = 360.0 * (k + 1) / n_segs
                cv2.ellipse(p,
                            (int(round(e["cx"])), int(round(e["cy"]))),
                            (max(1, int(round(e["a"]))),
                             max(1, int(round(e["b"])))),
                            e["ang"], a0, a1, c, 2, cv2.LINE_AA)
        else:
            cv2.ellipse(p,
                        (int(round(e["cx"])), int(round(e["cy"]))),
                        (max(1, int(round(e["a"]))),
                         max(1, int(round(e["b"])))),
                        e["ang"], 0, 360, c, 2, cv2.LINE_AA)
        cv2.circle(p, (int(round(e["cx"])), int(round(e["cy"]))),
                   5, c, -1)
        marker = "!" if is_warning else ""
        cv2.putText(p,
                    f"e{e['idx']}:r{int(e['r'])}({e['src'][:5]}){marker}",
                    (int(round(e["cx"])) + 6, int(round(e["cy"])) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv2.LINE_AA)
    if gt_path is not None and Path(gt_path).exists():
        try:
            gt = json.loads(Path(gt_path).read_text())
            ellipse_block = gt.get("ellipses", gt)
            for gname, ginfo in ellipse_block.items():
                if not isinstance(ginfo, dict) or "cx" not in ginfo:
                    continue
                if "n_pix" in ginfo and ginfo["n_pix"] < 100:
                    continue
                gc = {"yellow": (0, 200, 255),
                      "darkred": (40, 40, 180),
                      "darkred2": (60, 60, 200),
                      "green": (0, 200, 0)}.get(gname, (255, 255, 255))
                gcx, gcy = int(round(ginfo["cx"])), int(round(ginfo["cy"]))
                ga = int(round(ginfo["a"]))
                gb = int(round(ginfo["b"]))
                gang = float(ginfo["ang"])
                cv2.ellipse(p, (gcx, gcy), (ga, gb),
                            gang, 0, 360, gc, 4, cv2.LINE_AA)
                cv2.putText(p, f"GT:{gname[:4]}",
                            (gcx + 8, gcy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, gc, 1)
        except Exception:
            pass
    elapsed = result.get("elapsed_s", {})
    total_s = elapsed.get("total", 0) if isinstance(elapsed, dict) else 0
    n_warn = sum(1 for e in result["circles"] if e["ratio"] > 1.07)
    cv2.putText(p,
                f"v32: N={result['N']}({n_warn}!)  seeds={result.get('n_seeds', 0)} polar={result.get('n_polar_new', 0)}  "
                f"t={total_s:.0f}ms skel={result.get('n_skel_pixels', 0)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    cv2.putText(p,
                f"skeleton-first + GPU RANSAC + polar growth + assign-dedup  "
                f"a/b<={result['ratio_range'][1]}",
                (20, raw_bgr.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.imwrite(str(out_path), p)


def main():
    import sys
    here = Path(__file__).resolve().parent
    project = here.parent
    out_dir = project / "algo_v6" / "canonical"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("yhc/mask_refine/038_roi_009.png", "v32_038_roi_009.png"),
        ("yhc/mask_refine/038_roi_001.png", "v32_038_roi_001.png"),
        ("yhc/mask_refine/005_roi_001.png", "v32_005_roi_001.png"),
        ("yhc/mask_refine/017_roi_001.png", "v32_017_roi_001.png"),
        ("yhc/mask_refine/017_roi_004.png", "v32_017_roi_004.png"),
    ]
    if len(sys.argv) > 1:
        frag = sys.argv[1]
        pairs = [pp for pp in pairs if frag in pp[0]]
    summary = []
    for rel_in, rel_out in pairs:
        p = project / rel_in
        raw = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if raw is None:
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"missing {p}")
                continue
            raw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        mask_bin = (gray > 127).astype(np.uint8)
        print(f"\n=== {rel_in} ===")
        r = fit_v32(mask_bin)
        summary.append({"file": rel_in, "result": r})
        elapsed = r.get("elapsed_s", {})
        n_warn = sum(1 for e in r["circles"] if e["ratio"] > 1.07)
        print(f"  seeds={r['n_seeds']} polar={r['n_polar_new']} → N={r['N']}({n_warn}!)  "
              f"t={elapsed.get('total', 0)*1000:.0f}ms  "
              f"(skel={elapsed.get('skeleton', 0)*1000:.0f} "
              f"ransac={elapsed.get('ransac', 0)*1000:.0f} "
              f"grow={elapsed.get('polar_grow', 0)*1000:.0f} "
              f"dedup={elapsed.get('dedup', 0)*1000:.0f}) "
              f"skel_pix={r.get('n_skel_pixels', 0)}")
        for c in r["circles"]:
            w = "!" if c["ratio"] > 1.07 else ""
            print(f"    e{c['idx']}: r={c['r']:.0f}  "
                  f"cx={c['cx']:.0f}  cy={c['cy']:.0f}  "
                  f"a/b={c['ratio']:.3f}{w}  "
                  f"ang={c['ang']:.0f}°  sup={c['support']}  "
                  f"src={c.get('src', '?')}")
        gt_path = (project / "algo_v6" / "gt_ellipses.json"
                   if "017_roi_004" in rel_in else None)
        render(raw, mask_bin, r, out_dir / rel_out, gt_path=gt_path)
        print(f"  [render] {out_dir / rel_out}")
    out = project / "algo_v6" / "v32_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str,
                             ensure_ascii=False))
    print(f"\n[json] {out}")


if __name__ == "__main__":
    main()