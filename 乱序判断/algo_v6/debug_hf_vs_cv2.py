"""Debug HF vs cv2 on REAL v39 inlier sets."""
import sys, time
from pathlib import Path
import numpy as np, cv2, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v32, v35, v38, v39, ellipse_halir_flusser as hf

project = Path(__file__).resolve().parent.parent
MASKS_DIR = project / "yhc" / "mask_refine"

n_hf_none = 0; n_cv_none = 0; n_both_ok = 0
a_diffs = []; b_diffs = []; cx_diffs = []; cy_diffs = []
results = []

for f in sorted(MASKS_DIR.glob("*.png"))[:20]:
    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    mask_bin = (img > 127).astype(np.uint8)
    H, W = mask_bin.shape[:2]
    skel, skel_pixels = v32._skeleton(mask_bin)
    if len(skel_pixels) < 30: continue
    ys0, xs0 = np.where(mask_bin > 0)
    cx_est = float(xs0.mean()); cy_est = float(ys0.mean())
    r_max_est = max(20.0, np.hypot(xs0.max() - cx_est, ys0.max() - cy_est))
    r_min_est = max(20.0, r_max_est * 0.10)
    r_max = r_max_est * 0.95
    rng = np.random.default_rng(42)
    seeds_for_debug = []
    claimed = np.zeros(len(skel_pixels), dtype=bool)
    for _ in range(12):
        free = skel_pixels[~claimed]
        if len(free) < 30: break
        res = v35._ransac_all_rounds_gpu(
            free, n_rounds=1, n_iter=200, rng=rng,
            r_min=r_min_est, r_max=r_max, band_px=5.0, min_support=80)
        if not res: break
        _, cx0, cy0, r0, inliers, score, sigma = res[0]
        free_idx = np.where(~claimed)[0]
        claimed[free_idx[inliers]] = True
        if sigma > 8.0: continue
        d = np.sqrt((free[:,0]-cx0)**2 + (free[:,1]-cy0)**2)
        new_mask = np.abs(d - r0) < (3.0 * sigma)
        if new_mask.sum() < 80: continue
        pts = free[new_mask]
        # cv2
        ref_cv = v32._fit_ellipse_safe(pts.reshape(-1, 1, 2))
        # HF
        ref_hf = hf.halir_flusser_fit(pts)
        # Compare
        cv_ok = ref_cv is not None
        hf_ok = ref_hf is not None
        if not cv_ok: n_cv_none += 1
        if not hf_ok: n_hf_none += 1
        if cv_ok and hf_ok:
            n_both_ok += 1
            cx_diffs.append(abs(ref_cv[0] - ref_hf[0]))
            cy_diffs.append(abs(ref_cv[1] - ref_hf[1]))
            a_diffs.append(abs(ref_cv[2] - ref_hf[2]))
            b_diffs.append(abs(ref_cv[3] - ref_hf[3]))
            seeds_for_debug.append({"file": f.name, "r0": r0, "n_pts": len(pts),
                                    "cv": ref_cv, "hf": ref_hf})
        elif cv_ok and not hf_ok:
            # CV succeeds, HF fails: most informative
            seeds_for_debug.append({"file": f.name, "r0": r0, "n_pts": len(pts),
                                    "cv": ref_cv, "hf": None})
        # Claim
        dx = skel_pixels[:,0]-cx0; dy = skel_pixels[:,1]-cy0
        ds = np.sqrt(dx*dx+dy*dy)
        claimed = claimed | (np.abs(ds - r0) <= 5.0)
    results.append({"file": f.name, "seeds": seeds_for_debug})

print(f"20-image debug:")
print(f"  CV succeeded:        {sum(1 for r in results for s in r['seeds'] if s['cv'] is not None)}")
print(f"  HF succeeded:        {sum(1 for r in results for s in r['seeds'] if s['hf'] is not None)}")
print(f"  CV only (HF failed): {sum(1 for r in results for s in r['seeds'] if s['cv'] is not None and s['hf'] is None)}")
print(f"  HF only (CV failed): {sum(1 for r in results for s in r['seeds'] if s['hf'] is not None and s['cv'] is None)}")
print(f"  Both succeeded:      {n_both_ok}")

# Check ratios when HF fails but CV succeeds
fail_examples = [s for r in results for s in r['seeds'] if s['cv'] is not None and s['hf'] is None]
print(f"\nFirst 5 CV-success / HF-fail cases:")
for s in fail_examples[:5]:
    print(f"  {s['file']} r0={s['r0']:.1f}  n_pts={s['n_pts']}  "
          f"cv=({s['cv'][0]:.1f},{s['cv'][1]:.1f}, a={s['cv'][2]:.1f}, b={s['cv'][3]:.1f}, "
          f"ratio={s['cv'][2]/s['cv'][3]:.4f})")

# Stats on differences where both OK
if a_diffs:
    a = np.array(a_diffs); b = np.array(b_diffs)
    cx = np.array(cx_diffs); cy = np.array(cy_diffs)
    print(f"\nWhen both succeed (n={n_both_ok}):")
    print(f"  |a_diff|  med={np.median(a):.3f}  max={a.max():.3f}")
    print(f"  |b_diff|  med={np.median(b):.3f}  max={b.max():.3f}")
    print(f"  |cx_diff| med={np.median(cx):.3f}  max={cx.max():.3f}")
    print(f"  |cy_diff| med={np.median(cy):.3f}  max={cy.max():.3f}")