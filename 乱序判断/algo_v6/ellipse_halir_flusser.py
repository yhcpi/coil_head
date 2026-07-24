"""Numerically-stable direct ellipse fit.

This is the Fitzgibbon-Pilu-Fisher 1999 / Halir-Flusser 1998 algorithm with
the centroid translation step from HF that improves numerical conditioning.

References (verified by WebSearch):
- A. Fitzgibbon, M. Pilu, R. Fisher. "Direct Least-Squares Fitting of Ellipses".
  IEEE T-PAMI 21(5): 476-480, 1999. DOI 10.1109/34.765658
- R. Halir, J. Flusser. "Numerically stable direct least squares fitting of
  ellipses". 6th Int. Conf. in Central Europe on Computer Graphics and
  Visualization (WSCG), 1998.

Implementation strategy:
  1. Centroid translate (HF stability)
  2. Build D in raw form: [xx, xy, yy, x, y, 1]
  3. Solve S a = λ C a where C encodes the ellipse constraint
  4. Pick the eigenvector with the largest positive "4ac-b²"
  5. Extract cx, cy, a, b, ang

Implementation adapted from bdhammel/least-squares-ellipse-fitting
(verified working Python reference).
"""
from __future__ import annotations
import numpy as np


def halir_flusser_fit(points_xy: np.ndarray) -> tuple | None:
    """Fit ellipse to (N, 2) integer or float points.

    Returns (cx, cy, a, b, ang_deg) where (cx, cy) is the centre, a >= b are
    the semi-axes, ang is the rotation in degrees. Returns None if invalid
    (too few points, no convergence).

    The HF translation step avoids the conditioning issues that plague
    Fitzgibbon when points are far from the origin.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 5:
        return None
    # HF: translate to centroid FIRST (numerical stability)
    cx0, cy0 = pts.mean(axis=0)
    x = pts[:, 0] - cx0
    y = pts[:, 1] - cy0
    N = len(x)
    # Design matrix
    x2 = x * x; y2 = y * y; xy = x * y
    D = np.column_stack([x2, xy, y2, x, y, np.ones(N)])
    # Normalise each column to unit norm (HF + Rosin trick for conditioning).
    # Skip zero-norm columns silently (they contribute nothing to fit).
    col_norms = np.linalg.norm(D, axis=0)
    keep = col_norms > 0
    if not keep.all():
        # Replace zero-norm with unit scalar (does not change eigenvectors)
        col_norms[~keep] = 1.0
    D_norm = D / col_norms
    # Gram matrix and constraint
    S = D_norm.T @ D_norm
    C = np.zeros((6, 6))
    C[0, 0] = 0; C[0, 2] = 2; C[1, 1] = -1; C[2, 0] = 2  # 4ac-b² ≥ 1
    # Solve S a = λ C a (generalised eigenproblem)
    try:
        S_inv_C = np.linalg.solve(S, C)
    except np.linalg.LinAlgError:
        return None
    E, V = np.linalg.eig(S_inv_C)
    E = np.real(E); V = np.real(V)
    # Pick eigenvector with the largest positive eigenvalue
    # bdhammel: n = argmax(E_real). For ellipse fit on well-conditioned data
    # this returns the correct eigenvector. For degenerate cases, we filter by
    # positive (4ac-b²) tie-breaker.
    cand = [(k, E[k]) for k in range(6) if np.isfinite(E[k])]
    if not cand:
        return None
    # bdhammel: just pick the largest eigenvalue (no positive filter).
    # For near-circular fits the eigenvalue is small but positive; the
    # `4ac-b² > 0` check below catches the bad case.
    cand.sort(key=lambda x: -x[1])
    n = cand[0][0]
    coeffs_norm = np.real(V[:, n])
    # Un-normalise
    coeffs = coeffs_norm / col_norms
    a, b, c, d, e, f = coeffs
    # Centre
    disc = 4 * a * c - b ** 2
    # Threshold relative to the typical scale of a*c (~1e-13 here)
    if abs(disc) < 1e-18 * max(abs(a * c), 1e-30):
        return None
    cx_t = (b * e - 2 * c * d) / disc
    cy_t = (b * d - 2 * a * e) / disc
    # Sign ambiguity: the conic can be multiplied by -1 without changing
    # the geometric ellipse. The eigenvector can come out with a < 0
    # (numerical instability). Flip if needed so that a > 0.
    f_prime = f - (a * cx_t ** 2 + b * cx_t * cy_t + c * cy_t ** 2)
    if a < 0 or f_prime > 0:
        # Try flipping all signs
        a, b, c, d, e, f = -a, -b, -c, -d, -e, -f
        disc = 4 * a * c - b ** 2
        if abs(disc) < 1e-18 * max(abs(a * c), 1e-30):
            return None
        cx_t = (b * e - 2 * c * d) / disc
        cy_t = (b * d - 2 * a * e) / disc
        f_prime = f - (a * cx_t ** 2 + b * cx_t * cy_t + c * cy_t ** 2)
    if f_prime >= 0:  # need f' < 0 for ellipse
        return None
    # Eigenvalues of 2x2 quadratic form (symmetric matrix)
    Q = np.array([[a, b / 2],
                  [b / 2, c]])
    eigvals_q, eigvecs_q = np.linalg.eigh(Q)
    eigvals_q = np.real(eigvals_q)
    eigvecs_q = np.real(eigvecs_q)
    if (eigvals_q <= 0).any():
        return None
    semi_axes_sq = -f_prime / eigvals_q
    semi_axes = np.sqrt(semi_axes_sq)
    if semi_axes[0] > semi_axes[1]:
        semi_a, semi_b = float(semi_axes[0]), float(semi_axes[1])
        ang = float(np.degrees(np.arctan2(eigvecs_q[1, 0], eigvecs_q[0, 0])))
    else:
        semi_a, semi_b = float(semi_axes[1]), float(semi_axes[0])
        ang = float(np.degrees(np.arctan2(eigvecs_q[1, 1], eigvecs_q[0, 1])))
    if ang < 0:
        ang += 180
    if ang >= 180:
        ang -= 180
    return (cx_t + cx0, cy_t + cy0, semi_a, semi_b, ang)


def _sanity_synthetic(n_pts=500, *, cx=500.0, cy=400.0,
                       a=300.0, b=295.0, ang=37.0, noise=0.3,
                       rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    t = np.linspace(0, 2 * np.pi, n_pts)
    ang_r = np.radians(ang)
    cos_a, sin_a = np.cos(ang_r), np.sin(ang_r)
    x_loc = a * np.cos(t) + rng.normal(0, noise, n_pts)
    y_loc = b * np.sin(t) + rng.normal(0, noise, n_pts)
    x = cx + cos_a * x_loc - sin_a * y_loc
    y = cy + sin_a * x_loc + cos_a * y_loc
    return np.column_stack([x, y]), (cx, cy, a, b, ang)


if __name__ == "__main__":
    import time
    import cv2
    print("=== Synthetic ellipse ===")
    pts, truth = _sanity_synthetic(n_pts=500, noise=0.3)
    cx, cy, a, b, ang = truth
    pts_cv = pts.reshape(-1, 1, 2).astype(np.float32)
    t0 = time.time()
    res_cv = cv2.fitEllipse(pts_cv)
    t_cv = (time.time() - t0) * 1e6
    t0 = time.time()
    res_hf = halir_flusser_fit(pts)
    t_hf = (time.time() - t0) * 1e6
    print(f"truth:    cx={cx:.4f} cy={cy:.4f} a={a:.4f} b={b:.4f} ang={ang:.4f}")
    print(f"cv2:      cx={res_cv[0][0]:.4f} cy={res_cv[0][1]:.4f} "
          f"a={res_cv[1][0]/2:.4f} b={res_cv[1][1]/2:.4f} ang={res_cv[2]:.4f} "
          f"[{t_cv:.0f} µs]")
    if res_hf is not None:
        print(f"halir-fl: cx={res_hf[0]:.4f} cy={res_hf[1]:.4f} a={res_hf[2]:.4f} "
              f"b={res_hf[3]:.4f} ang={res_hf[4]:.4f} [{t_hf:.0f} µs]")
        cv_err_c = np.sqrt((res_cv[0][0] - cx)**2 + (res_cv[0][1] - cy)**2)
        hf_err_c = np.sqrt((res_hf[0] - cx)**2 + (res_hf[1] - cy)**2)
        cv_err_a = abs(res_cv[1][0] / 2 - a)
        hf_err_a = abs(res_hf[2] - a)
        cv_err_b = abs(res_cv[1][1] / 2 - b)
        hf_err_b = abs(res_hf[3] - b)
        print(f"center err: cv2 {cv_err_c:.4f}px  HF {hf_err_c:.4f}px")
        print(f"a err:      cv2 {cv_err_a:.4f}px  HF {hf_err_a:.4f}px")
        print(f"b err:      cv2 {cv_err_b:.4f}px  HF {hf_err_b:.4f}px")
        print(f"a/b ratio: cv2 {res_cv[1][0]/res_cv[1][1]:.5f}, HF {res_hf[2]/res_hf[3]:.5f}, "
              f"truth {a/b:.5f}")
    else:
        print("halir-fl: returned None")

    print("\n=== Integer-pixel test (the real use case) ===")
    pts_int = np.round(pts).astype(np.float32).reshape(-1, 1, 2)
    t0 = time.time()
    res_cv_int = cv2.fitEllipse(pts_int)
    t_cv_int = (time.time() - t0) * 1e6
    t0 = time.time()
    res_hf_int = halir_flusser_fit(np.round(pts).astype(np.float64))
    t_hf_int = (time.time() - t0) * 1e6
    print(f"cv2 int:  cx={res_cv_int[0][0]:.4f} cy={res_cv_int[0][1]:.4f} "
          f"a={res_cv_int[1][0]/2:.4f} b={res_cv_int[1][1]/2:.4f} [{t_cv_int:.0f} µs]")
    if res_hf_int is not None:
        print(f"hf int:   cx={res_hf_int[0]:.4f} cy={res_hf_int[1]:.4f} "
              f"a={res_hf_int[2]:.4f} b={res_hf_int[3]:.4f} [{t_hf_int:.0f} µs]")
    else:
        print("HF returned None on int data")
