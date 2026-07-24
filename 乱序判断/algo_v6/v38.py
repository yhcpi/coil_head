"""v38: v35 + support/r ratio-driven adaptive min_arc_coverage.

Why support/r is a better adaptive signal than σ:
  - σ ∈ [1.0, 2.2] in real data → only 2× spread, no discrimination
  - support/r ∈ [1.5, 4.6] in real data → 5× spread
  - support/r < 2 → sparse inliers per unit r → likely partial arc
  - support/r > 4 → dense inliers per unit r → clearly a real full circle

The arc_min threshold maps support_per_r → threshold:
  support/r high (4+)  → arc_min low (0.15) → trust dense circle
  support/r medium (2-4) → arc_min medium (0.25) → standard
  support/r low (< 1.5) → arc_min high (0.45) → drop sparse/partial

Reference: classic robust estimator pattern; linear interpolation is a
standard way to make one signal modulate another threshold.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import v32
import v35

PALETTE = v32.PALETTE


def adaptive_arc_min_support(support_per_r, *,
                             low_sr=1.5, high_sr=4.0,
                             arc_min_hi=0.45, arc_min_lo=0.15):
    """Adaptive min_arc_coverage driven by support/r ratio.

    Args:
      support_per_r: inliers per unit r
      low_sr: support/r below this → arc_min_hi (strict)
      high_sr: support/r above this → arc_min_lo (loose)
      arc_min_hi/lo: arc coverage thresholds for sparse / dense circles
    """
    if support_per_r <= low_sr:
        return arc_min_hi
    if support_per_r >= high_sr:
        return arc_min_lo
    frac = (support_per_r - low_sr) / (high_sr - low_sr)
    return arc_min_hi - frac * (arc_min_hi - arc_min_lo)


def fit_v38(mask_bin, *,
            n_rounds=12, n_iter=300,
            band_px=5.0, min_support=80,
            r_min_factor=0.10, r_max_factor=0.95,
            ratio_min=1.00, ratio_max=1.10,
            support_min=400, dr_tol_px=5.0,
            band_px_init=8.0, k_sigma=3.0, sigma_max=8.0,
            # support/r adaptive arc params
            arc_lo_sr=1.5, arc_hi_sr=4.0,
            arc_min_hi=0.45, arc_min_lo=0.15,
            fast_mode=False, rng_seed=42) -> dict[str, Any]:
    """v38: v35 + support/r ratio-driven adaptive min_arc_coverage."""
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
        # Adaptive min_arc_coverage via support/r
        support_per_r = score / r_ell if r_ell > 0 else 0
        arc_min = adaptive_arc_min_support(
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
            "src": "v38_support_adaptive",
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
    """v38 render: same as v35."""
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
        sr_str = f" sr{e.get('support_per_r', 0):.1f}/{e.get('arc_min', 0):.2f}"
        cv2.putText(p,
                    f"e{e['idx']}:r{int(e['r'])}{marker}{sr_str}",
                    (int(round(e["cx"])) + 6, int(round(e["cy"])) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, c, 1, cv2.LINE_AA)
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
                f"v38: N={result['N']}({n_warn}!)  t={total_s*1000:.0f}ms",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    cv2.putText(p,
                f"support/r-driven arc coverage  a/b<={result['ratio_range'][1]}",
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
        ("yhc/mask_refine/038_roi_009.png", "v38_038_roi_009.png"),
        ("yhc/mask_refine/038_roi_001.png", "v38_038_roi_001.png"),
        ("yhc/mask_refine/005_roi_001.png", "v38_005_roi_001.png"),
        ("yhc/mask_refine/017_roi_001.png", "v38_017_roi_001.png"),
        ("yhc/mask_refine/017_roi_004.png", "v38_017_roi_004.png"),
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
        r = fit_v38(mask_bin)
        summary.append({"file": rel_in, "result": r})
        elapsed = r.get("elapsed_s", {})
        n_warn = sum(1 for e in r["circles"] if e["ratio"] > 1.07)
        print(f"  → N={r['N']}({n_warn}!)  t={elapsed.get('total', 0)*1000:.0f}ms")
        for c in r["circles"]:
            w = "!" if c["ratio"] > 1.07 else ""
            print(f"    e{c['idx']}: r={c['r']:.0f}  a/b={c['ratio']:.3f}{w}  "
                  f"sr={c.get('support_per_r', 0):.2f}  arc_min={c.get('arc_min', 0):.2f}  "
                  f"σ={c.get('sigma', 0):.1f}  sup={c['support']}")
        gt_path = (project / "algo_v6" / "gt_ellipses.json"
                   if "017_roi_004" in rel_in else None)
        render(raw, mask_bin, r, out_dir / rel_out, gt_path=gt_path)
        print(f"  [render] {out_dir / rel_out}")
    out = project / "algo_v6" / "v38_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str,
                             ensure_ascii=False))
    print(f"\n[json] {out}")


if __name__ == "__main__":
    main()