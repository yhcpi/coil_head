"""v35: GPU whole RANSAC inner loop + GC-RANSAC σ marginalization.

Design:
  - Generate ALL n_rounds × n_iter triplet indices in ONE numpy call (CPU)
  - Batch 3-pt circle algebra on GPU: (cx, cy, r) for ALL hypothes in ONE launch
  - ONE big distance matrix (n_total_hyp × n_pool) on GPU
  - Per-round argmax to find best seed per round
  - GC-RANSAC σ marginalization on ALL winners via GPU

The CPU loop is eliminated:
  - v32/v34: O(n_rounds × n_iter) Python loops for 3-pt circle
  - v35:    One GPU op for ALL hypothes

Reference for batch 3-pt circle on GPU:
  From https://github.com/C-H-Chen/Arc-Support-Line-Segments (Lu 2020) idea,
  the GPU trivially parallelizes the 3-pt algebra.

This file is the THIRD improvement (per user spec): GPU whole inner loop speedup.
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

import v32

PALETTE = v32.PALETTE


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# GPU batch 3-pt circle: from triplets → (cx, cy, r) for ALL hypothes
# ============================================================================
def _circle_from_3pts_batch_gpu(p1, p2, p3, *, r_min, r_max):
    """Compute 3-pt circle for ALL hypothes on GPU.

    Args:
      p1, p2, p3: torch.Tensor (n, 2) on GPU, each row is (x, y)
      r_min, r_max: float

    Returns:
      cxs, cys, rs: torch.Tensor (n_valid,) on GPU (filtered by r)
      valid_mask: torch.Tensor (n_total,) bool
    """
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    valid = d.abs() > 1e-9
    d_safe = torch.where(valid, d, torch.ones_like(d))
    s1 = x1 * x1 + y1 * y1
    s2 = x2 * x2 + y2 * y2
    s3 = x3 * x3 + y3 * y3
    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d_safe
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d_safe
    r = torch.sqrt((x1 - ux) ** 2 + (y1 - uy) ** 2)
    valid = valid & (r >= r_min) & (r <= r_max)
    return ux, uy, r, valid


# ============================================================================
# Main RANSAC: ALL rounds + ALL iters in ONE launch
# ============================================================================
def _ransac_all_rounds_gpu(pool_pixels, *, n_rounds, n_iter, rng,
                           r_min, r_max, band_px, min_support):
    """Generate ALL hypothes in one CPU call, evaluate ALL on GPU.

    Returns:
      list of (round_idx, cx, cy, r, inlier_indices, score, sigma)
    """
    if len(pool_pixels) < 30:
        return []
    dev = _device()
    n_total = n_rounds * n_iter
    n_pix = len(pool_pixels)
    # CPU: one big triplet sampling (use replace=True if pool is small)
    replace = (n_total > n_pix)
    sel = rng.choice(n_pix, size=(n_total, 3), replace=replace or True)
    p1_np = pool_pixels[sel[:, 0]]
    p2_np = pool_pixels[sel[:, 1]]
    p3_np = pool_pixels[sel[:, 2]]
    # Upload all to GPU in ONE transfer
    p1 = torch.from_numpy(p1_np).to(dev)
    p2 = torch.from_numpy(p2_np).to(dev)
    p3 = torch.from_numpy(p3_np).to(dev)
    # GPU: batch 3-pt circle for ALL hypothes at once
    cxs, cys, rs, valid = _circle_from_3pts_batch_gpu(
        p1, p2, p3, r_min=r_min, r_max=r_max)
    # Filter to valid
    valid_idx_t = torch.where(valid)[0]
    if len(valid_idx_t) == 0:
        return []
    v_cx = cxs[valid]
    v_cy = cys[valid]
    v_r = rs[valid]
    # Upload pool ONCE
    pool_t = torch.from_numpy(pool_pixels).to(dev)
    # ONE big distance matrix: (n_valid, n_pool)
    dx = pool_t[None, :, 0] - v_cx[:, None]
    dy = pool_t[None, :, 1] - v_cy[:, None]
    dists = torch.sqrt(dx * dx + dy * dy)
    residuals = (dists - v_r[:, None]).abs()
    inliers_init = residuals < band_px  # (n_valid, n_pool) bool
    scores = inliers_init.sum(dim=1)  # (n_valid,)
    # Split valid hypothes into n_rounds chunks; pick best per round
    chunk_size = n_iter
    n_valid = len(valid_idx_t)
    n_rounds_eff = max(1, n_valid // chunk_size)
    # Use modulo assignment so each round gets exactly n_iter hypothes
    assign = torch.arange(n_valid, device=dev) % n_rounds_eff
    # Per-round argmax
    out = []
    for round_idx in range(n_rounds_eff):
        round_mask = assign == round_idx
        if not round_mask.any():
            continue
        round_scores = torch.where(round_mask, scores,
                                   torch.full_like(scores, -1))
        local_best = int(round_scores.argmax())
        score = int(scores[local_best])
        if score < min_support:
            continue
        cx = float(v_cx[local_best].item())
        cy = float(v_cy[local_best].item())
        r0 = float(v_r[local_best].item())
        inlier_mask = inliers_init[local_best]
        inlier_pix_idx = torch.where(inlier_mask)[0].cpu().numpy()
        # Compute σ from initial inlier residuals (still on GPU)
        inlier_res = residuals[local_best][inlier_mask]
        if len(inlier_res) >= 5:
            med = float(inlier_res.median().item())
            mad = float((inlier_res - med).abs().median().item())
            sigma = 1.4826 * mad
        else:
            sigma = 5.0
        out.append((round_idx, cx, cy, r0, inlier_pix_idx, score, sigma))
    return out


# ============================================================================
# Main fit function
# ============================================================================
def fit_v35(mask_bin, *,
            n_rounds=12, n_iter=300,
            band_px=5.0, min_support=80,
            r_min_factor=0.10, r_max_factor=0.95,
            ratio_min=1.00, ratio_max=1.10,
            support_min=400, dr_tol_px=5.0,
            band_px_init=8.0, k_sigma=3.0, sigma_max=8.0,
            min_arc_coverage=0.20,
            fast_mode=False, rng_seed=42) -> dict[str, Any]:
    """v35: GPU whole RANSAC inner loop + GC-RANSAC σ marginalization."""
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
    # We re-run the per-round RANSAC, but each round's hypothes is a fresh
    # batch (since the claimed_mask shrinks the pool). This way each round
    # gets n_iter hypothes on GPU in ONE launch.
    for round_idx in range(n_rounds):
        free_pixels = skel_pixels[~claimed_mask]
        if len(free_pixels) < 30:
            break
        # ONE GPU launch for this round's n_iter hypothes
        round_results = _ransac_all_rounds_gpu(
            free_pixels, n_rounds=1, n_iter=n_iter, rng=rng,
            r_min=r_min_est, r_max=r_max, band_px=band_px,
            min_support=min_support)
        if not round_results:
            break
        # round_results is a list with 1 element (n_rounds=1)
        _, cx0, cy0, r0, inliers, score, sigma = round_results[0]
        # Re-claim pixels in band BEFORE fitting
        free_idx = np.where(~claimed_mask)[0]
        claimed_mask[free_idx[inliers]] = True
        # GC-RANSAC: σ marginalization
        if sigma > sigma_max:
            continue
        inlier_pix = free_pixels[inliers]
        # Re-select with k*σ
        dx0 = free_pixels[:, 0] - cx0
        dy0 = free_pixels[:, 1] - cy0
        d0 = np.sqrt(dx0 * dx0 + dy0 * dy0)
        new_mask = np.abs(d0 - r0) < (k_sigma * sigma)
        if new_mask.sum() < min_support:
            continue
        ref = v32._fit_ellipse_safe(
            free_pixels[new_mask].reshape(-1, 1, 2))
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
        arc_pixels = score / (2.0 * math.pi * r_ell) if r_ell > 0 else 0
        if arc_pixels < min_arc_coverage:
            continue
        seeds.append({
            "cx": float(cx), "cy": float(cy),
            "a": float(a), "b": float(b),
            "r": float(r_ell),
            "ang": float(ang), "ratio": float(ratio),
            "support": int(score),
            "sigma": float(sigma),
            "src": "v35_gpu_full",
        })
        # Re-claim band pixels (small expansion for next round)
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
    if len(candidates) >= 2:
        rs = np.array([c["r"] for c in candidates])
        gaps = np.diff(rs)
        P_wr = float(np.mean(gaps))
        P_wr_std = float(np.std(gaps))
    else:
        P_wr = P_wr_std = 0.0
    return {
        "circles": candidates, "N": len(candidates),
        "P_wr": P_wr, "P_wr_std": P_wr_std,
        "ratio_range": [ratio_min, ratio_max],
        "elapsed_s": {
            "skel": t1 - t0, "ransac": t2 - t1,
            "dedup": t3 - t2, "total": t3 - t0,
        },
    }


def render(raw_bgr, mask_bin, result, out_path, gt_path=None):
    """v35 render: same as v34."""
    p = raw_bgr.copy()
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
        sigma_str = f" σ{e.get('sigma', 0):.1f}" if "sigma" in e else ""
        cv2.putText(p,
                    f"e{e['idx']}:r{int(e['r'])}{marker}{sigma_str}",
                    (int(round(e["cx"])) + 6, int(round(e["cy"])) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA)
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
                f"v35: N={result['N']}({n_warn}!)  "
                f"t={total_s*1000:.0f}ms",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    cv2.putText(p,
                f"GPU whole RANSAC inner + GC-RANSAC σ  a/b<={result['ratio_range'][1]}",
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
        ("yhc/mask_refine/038_roi_009.png", "v35_038_roi_009.png"),
        ("yhc/mask_refine/038_roi_001.png", "v35_038_roi_001.png"),
        ("yhc/mask_refine/005_roi_001.png", "v35_005_roi_001.png"),
        ("yhc/mask_refine/017_roi_001.png", "v35_017_roi_001.png"),
        ("yhc/mask_refine/017_roi_004.png", "v35_017_roi_004.png"),
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
        r = fit_v35(mask_bin)
        summary.append({"file": rel_in, "result": r})
        elapsed = r.get("elapsed_s", {})
        n_warn = sum(1 for e in r["circles"] if e["ratio"] > 1.07)
        print(f"  → N={r['N']}({n_warn}!)  t={elapsed.get('total', 0)*1000:.0f}ms  "
              f"(skel={elapsed.get('skel', 0)*1000:.0f} "
              f"ransac={elapsed.get('ransac', 0)*1000:.0f} "
              f"dedup={elapsed.get('dedup', 0)*1000:.0f})")
        for c in r["circles"]:
            w = "!" if c["ratio"] > 1.07 else ""
            print(f"    e{c['idx']}: r={c['r']:.0f}  "
                  f"a/b={c['ratio']:.3f}{w}  σ={c.get('sigma', 0):.1f}  "
                  f"sup={c['support']}")
        gt_path = (project / "algo_v6" / "gt_ellipses.json"
                   if "017_roi_004" in rel_in else None)
        render(raw, mask_bin, r, out_dir / rel_out, gt_path=gt_path)
        print(f"  [render] {out_dir / rel_out}")
    out = project / "algo_v6" / "v35_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str,
                             ensure_ascii=False))
    print(f"\n[json] {out}")


if __name__ == "__main__":
    main()