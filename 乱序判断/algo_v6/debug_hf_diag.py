"""Diagnose why HF returns None on near-circular real data."""
import sys
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v32, v35, ellipse_halir_flusser as hf

project = Path(__file__).resolve().parent.parent
MASKS_DIR = project / "yhc" / "mask_refine"

# Take the first image, run v39 pipeline to first inlier set
f = sorted(MASKS_DIR.glob("*.png"))[5]  # 005_roi_001 likely
img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
mask_bin = (img > 127).astype(np.uint8)
H, W = mask_bin.shape[:2]
skel, skel_pixels = v32._skeleton(mask_bin)
ys0, xs0 = np.where(mask_bin > 0)
cx_est = float(xs0.mean()); cy_est = float(ys0.mean())
r_max_est = max(20.0, np.hypot(xs0.max() - cx_est, ys0.max() - cy_est))
r_min_est = max(20.0, r_max_est * 0.10)
r_max = r_max_est * 0.95
rng = np.random.default_rng(42)
res = v35._ransac_all_rounds_gpu(
    skel_pixels, n_rounds=1, n_iter=200, rng=rng,
    r_min=r_min_est, r_max=r_max, band_px=5.0, min_support=80)
_, cx0, cy0, r0, inliers, score, sigma = res[0]
d = np.sqrt((skel_pixels[:,0]-cx0)**2 + (skel_pixels[:,1]-cy0)**2)
new_mask = np.abs(d - r0) < (3.0 * sigma)
pts = skel_pixels[new_mask].astype(np.float64)
print(f"First inlier set from {f.name}: r0={r0:.1f}, n_pts={len(pts)}")
print(f"  pts range: x=[{pts[:,0].min():.0f},{pts[:,0].max():.0f}] "
      f"y=[{pts[:,1].min():.0f},{pts[:,1].max():.0f}]")
print(f"  centroid: ({pts[:,0].mean():.2f}, {pts[:,1].mean():.2f})")

# Run HF with internal tracing
def hf_traced(points):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 5:
        return None
    cx0_, cy0_ = pts.mean(axis=0)
    x = pts[:, 0] - cx0_; y = pts[:, 1] - cy0_
    N = len(x)
    x2 = x*x; y2 = y*y; xy = x*y
    D = np.column_stack([x2, xy, y2, x, y, np.ones(N)])
    col_norms = np.linalg.norm(D, axis=0)
    print(f"  col_norms: {col_norms}")
    keep = col_norms > 0
    cn = col_norms.copy()
    if not keep.all():
        cn[~keep] = 1.0
    D_norm = D / cn
    S = D_norm.T @ D_norm
    C = np.zeros((6,6))
    C[0, 0] = 0; C[0, 2] = 2; C[1, 1] = -1; C[2, 0] = 2
    S_inv_C = np.linalg.solve(S, C)
    E, V = np.linalg.eig(S_inv_C)
    E = np.real(E); V = np.real(V)
    print(f"  eigenvalues: {E}")
    cand = [(k, E[k]) for k in range(6) if np.isfinite(E[k])]
    cand.sort(key=lambda x: -x[1])
    n = cand[0][0]
    coeffs_norm = np.real(V[:, n])
    coeffs = coeffs_norm / cn
    a, b, c, d_, e_, f_ = coeffs
    print(f"  coeffs (a,b,c,d,e,f) = ({a:.3e}, {b:.3e}, {c:.3e}, {d_:.3e}, {e_:.3e}, {f_:.3e})")
    print(f"  4ac - b² = {4*a*c - b**2:.6e}   (>0 = good)")
    disc = 4 * a * c - b ** 2
    if abs(disc) < 1e-18 * max(abs(a * c), 1e-30):
        print(f"  >>> FAIL: disc too small"); return None
    cx_t = (b * e_ - 2 * c * d_) / disc
    cy_t = (b * d_ - 2 * a * e_) / disc
    f_prime = f_ - (a * cx_t ** 2 + b * cx_t * cy_t + c * cy_t ** 2)
    print(f"  f_prime = {f_prime:.6e}   (<0 = good)")
    if f_prime >= 0:
        print(f"  >>> FAIL: f_prime >= 0"); return None
    Q = np.array([[a, b/2], [b/2, c]])
    eigvals_q, eigvecs_q = np.linalg.eigh(Q)
    print(f"  Q eigvals: {eigvals_q}")
    if (eigvals_q <= 0).any():
        print(f"  >>> FAIL: Q eigvals <= 0"); return None
    semi_axes_sq = -f_prime / eigvals_q
    semi_axes = np.sqrt(semi_axes_sq)
    if semi_axes[0] > semi_axes[1]:
        sa, sb = float(semi_axes[0]), float(semi_axes[1])
        ang = float(np.degrees(np.arctan2(eigvecs_q[1,0], eigvecs_q[0,0])))
    else:
        sa, sb = float(semi_axes[1]), float(semi_axes[0])
        ang = float(np.degrees(np.arctan2(eigvecs_q[1,1], eigvecs_q[0,1])))
    return cx_t+cx0_, cy_t+cy0_, sa, sb, ang

print("\n=== HF traced on real inlier set ===")
res_hf = hf_traced(pts)
print(f"HF result: {res_hf}")
print(f"\ncv2 reference: {v32._fit_ellipse_safe(pts.reshape(-1, 1, 2))}")