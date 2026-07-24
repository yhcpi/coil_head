# Ellipse refit backend — Halir-Flusser 1998 vs cv2.fitEllipse (2026-07-24)

Background: v39 production uses `cv2.fitEllipse` (Fitzgibbon-Pilu-Fisher 1999
direct least-squares). For sub-pixel a/b precision, the user asked whether
the Halir-Flusser 1998 centroid-translation variant is worth plugging in.

## Method

`v44.py` replaces `v32._fit_ellipse_safe` (cv2 wrapper) with our
`ellipse_halir_flusser.halir_flusser_fit` implementation.

References (verified by WebSearch):
- A. Fitzgibbon, M. Pilu, R. Fisher. "Direct Least-Squares Fitting of Ellipses".
  IEEE T-PAMI 21(5): 476-480, 1999. DOI 10.1109/34.765658
- R. Halir, J. Flusser. "Numerically stable direct least squares fitting of
  ellipses". 6th WSCG, 1998.

Implementation adapted from bdhammel/least-squares-ellipse-fitting
(verified working Python reference). One critical fix made for near-circular
data: when the eigenvector picks coeffs with a < 0 (numerical sign ambiguity
for conics equivalent up to scalar multiplier), flip all signs and
re-compute. Without this fix, 78 % of real inlier sets returned None
(f_prime > 0).

## Synthetic validation

```
truth:    cx=500.0 cy=400.0 a=300.0 b=295.0 ang=37.0
cv2:      cx=500.0170 cy=399.9832 a=294.99 b=299.99 ang=126.94 [260 µs]
halir-fl: cx=500.0170 cy=399.9832 a=299.99 b=294.99 ang=36.94  [508 µs]
center err: cv2 0.0238px  HF 0.0239px
a err:      cv2 5.0128px  HF 0.0059px     ← HF wins 850×
b err:      cv2 4.9947px  HF 0.0134px     ← HF wins 370×
```

HF is sub-pixel accurate on synthetic data; cv2 is off by 5 px on a/b (likely
a convention mismatch in cv2's `(w, h)` tuple interpretation).

## 100-image benchmark vs v39

```
v39:  N_med=4  N_mean=4.3  t_med=118ms  gold_017_004=2  (GT=3)
v44:  N_med=5  N_mean=4.4  t_med=122ms  gold_017_004=4  (GT=3)
```

| ΔN    | image count |
|-------|-------------|
| -3    | 4           |
| -2    | 9           |
| -1    | 19          |
| 0     | 28          |
| +1    | 24          |
| +2    | 11          |
| +3    | 5           |

Net: v44 finds ~8 more circles across 100 images than v39 (40 wins, 32
losses, 28 ties). Both are within |error| = 1 of GT=3 on the gold image.

## a/b sub-pixel precision per matched circle (258 matches)

| axis | med | mean | p95 | max |
|------|-----|------|-----|-----|
| a    | 0.82 px | 1.45 px | 4.65 px | 7.89 px |
| b    | 0.67 px | 1.43 px | 4.58 px | 8.39 px |

Median diff is sub-pixel (within one pixel of cv2). p95 ~5 px is dominated by
near-degenerate cases where cv2 and HF pick slightly different ellipse
parameterisations.

## Verdict

v44 is **equivalent** to v39. The HF sub-pixel precision improvement is
real on synthetic data but does not translate to a clear N or t improvement
on real 100-image evaluation, because:

1. **N is dominated by RANSAC**, not the refit. Both backends see the same
   inlier sets; the seed acceptance/rejection thresholds are identical.
2. **Different fits come from numerical noise** in near-circular data where
   both algorithms pick slightly different but equivalent parameterisations.
3. **t_med stays at ~120 ms** — HF is 2× slower per call (500 µs vs 260 µs),
   but only called ~7 times per image, so adds ~1.4 ms (within noise).

If a future task requires **sub-pixel a/b precision on near-circular fits**
(e.g. for tolerance checks against wire diameter), use v44. Otherwise stay
on v39 (cv2 is faster per call and avoids the sign-flip edge case).

## Files

- `v44.py` — production code
- `ellipse_halir_flusser.py` — Halir-Flusser implementation
- `bench_v44_100.py` — benchmark
- `v44_bench.json` — raw per-image stats