# Algorithm chain (v32 → v35 → v38 → v39)

This document describes the **final production algorithm** (`algo_v6/v39.py`)
and the reasoning behind each component. Implementation files are kept in
`../algo_v6/` and are self-contained.

## Problem

Given a binary mask `M ∈ {0,1}^{H×W}` of a steel coil end-face, estimate
the number of distinct *wire turns* visible. Each wire turn is a near-circular
contour at radius `r` from the coil centre, with eccentricity (semi-axis ratio)
typically < 1.10 and perimetral count ~ 2π·r.

The mask is **noisy**: multiple turns may overlap locally, the centre of
curvature may not be exactly concentric, and a portion of every contour is
partially occluded by the next layer.

## Pipeline

```
binary mask
   │
   │  v32._skeleton(M)            Lee94 / Zhang-Suen medial axis
   ▼
skeleton pixels  S                H*Skel ≈ H·50 ms  (skimage)
   │
   │  v35._ransac_all_rounds_gpu  GPU batched 3-pt circle RANSAC
   │  12 rounds × 200 iter         + Gaussian-kernel inlier scoring
   ▼
seed circles  { (cx, cy, r, support, σ) }
   │
   │  v32._fit_ellipse_safe        cv2.fitEllipse on inliers ±k·σ
   ▼
ellipses  { (cx, cy, a, b, ang) }
   │
   │  v32._polar_grow + Gaussian-falloff refinement
   │  →  v38.adaptive_arc_min_support(support_per_r)
   ▼
filtered ellipses  (drop sparse / partial-arc candidates)
   │
   │  v32._r_cluster_dedup        keep highest-support per radius cluster
   ▼
final circles  N_t wire turns
```

Each stage is owned by one file:

| file          | responsibility                                |
|---------------|-----------------------------------------------|
| `v32.py`      | skeleton (Lee94), 3-pt-circle, fitter, dedup  |
| `v35.py`      | GPU-batched RANSAC inner loop                 |
| `v38.py`      | support/r-adaptive arc_min filter             |
| `v39.py`      | production (v35 + v38 + n_iter=200 budget)    |

## Stage 1 — Skeleton

```python
from skimage.morphology import skeletonize
def _skeleton(mask_bin):
    skel = skeletonize(mask_bin.astype(bool), method='lee')
    ys, xs = np.where(skel)
    return skel, np.column_stack((xs, ys))
```

We use **Lee 1994** (4-subiteration thinning, removes pixels except those with
exactly one 8-neighbour crossing) because it preserves connectivity where
Zhang-Suen 1984 may break thin junctions. On 1000×1000 coil masks this is the
fixed cost ≈ 50 ms; attempts to replace with `cv2.ximgproc.thinning`,
`skimage.skeletonize(method='zhang')` and a custom `torch.roll`-based GPU
Zhang-Suen all bottom out ≥ 60 ms (see `skel_torch.py`).

## Stage 2 — GPU-batch 3-pt-circle RANSAC (`v35`)

For each round:
1. Sample 200 triples `(p1, p2, p3)` from currently-`unclaimed` skeleton
   pixels (`replace`=True when the pool is < 600 pixels).
2. Compute the circumscribed circle of each triple — the **batched vectorised
   3-pt circle algebra** from

   > A. Fitzgibbon, M. Pilu, R. Fisher, *Direct Least-Squares Fitting of Ellipses*, IEEE TPAMI 1999. https://doi.org/10.1109/34.765658

   is reused for the per-triple circle centre and radius. In vectorised form:

   ```python
   a, b, c = p2 - p1, p3 - p1, p3 - p2
   d = 2 * (a.x*b.y - a.y*b.x)
   cx = ((b.y*(a.x**2 + a.y**2) - a.y*(b.x**2 + b.y**2)) / d) + p1.x
   cy = ((a.x*(b.x**2 + b.y**2) - b.x*(a.x**2 + a.y**2)) / d) + p1.y
   r = ||(cx, cy) - p1||
   ```

   then `score = Σ exp(-(d - r)² / (2·band²))` over all skeleton pixels
   evaluated via a GPU distance matrix.
3. Keep the best-scoring hypothesis; add its band-radius `±band_px` to the
   `claimed_mask` so the next round cannot re-detect the same wire turn.

We run 12 rounds. Each round sees fewer fresh skeleton pixels, which is the
*RANSAC-greedy* strategy of

> R. Schnabel, R. Klein, *Octree-based Fusion of RANSAC and Least-Squares Estimation*, 2007.

extended to GPU batch form. Total time 12×200 ≈ 100 ms on RTX 4060 Ti,
dominated by `torch.cdist` calls (≈ 5 ms per hypothesis batch of 200).

## Stage 3 — Adaptive arc_min via support/r (`v38`)

A naive constant `arc_min = 0.20` accepts circles whose inlier arc covers
20 % of the perimeter. On coil masks this admits too many **partial-arc**
false positives (small radius → easy to hit 20 % even on spurious corridors
in the skeleton).

**Key insight**: the *support-to-radius* ratio

  `support_per_r = inlier_pixel_count / r`

is a better discriminator than σ (inlier-noise standard deviation)
because the data spread is 5× wider:

|         | min    | max   | spread |
|---------|--------|-------|--------|
| σ       | 1.0    | 2.2   | 2.2×   |
| support/r | 1.5  | 4.6   | 3.1×   |

(Empirically measured on 100-image eval.)

We make `arc_min` a *decreasing* function of `support_per_r`:

```python
def adaptive_arc_min_support(support_per_r, *,
                             low_sr=1.5, high_sr=4.0,
                             arc_min_hi=0.45, arc_min_lo=0.15):
    if support_per_r <= low_sr: return arc_min_hi
    if support_per_r >= high_sr: return arc_min_lo
    f = (support_per_r - low_sr) / (high_sr - low_sr)
    return arc_min_hi - f * (arc_min_hi - arc_min_lo)
```

- `support/r = 1.0`: very sparse → `arc_min = 0.45` (reject partial)
- `support/r = 2.5`: typical → `arc_min = 0.30` (medium)
- `support/r = 4.0`: dense perimeter → `arc_min = 0.15` (allow)

This filter drops 95/100 images' circle counts (precision ↑) and **hits
ground truth exactly** on `017_roi_004` (N=3=GT, vs v35 N=6).

The intuition: a real wire turn at radius `r` has ≈ 2π·r inlier pixels;
when r is large, we naturally have more inliers; requiring a fixed 20 %
penalises small radii un-justly. Normalising by r cancels that.

## Stage 4 — Ellipse refit

For each accepted 3-pt-circle `(cx, cy, r)`, find all skeleton pixels
within `k·σ` band (default `k = 3.0`) and refit an ellipse via cv2.fitEllipse:

```python
(cxx, cyy, a, b, ang) = cv2.fitEllipse(points.reshape(-1, 1, 2))
```

Reject ellipses with `(a/b) > ratio_max = 1.10` (i.e. not circle-like).

## Stage 5 — r-cluster dedup

Group ellipses into bins of radius `±dr_tol_px = 5`. Within each cluster, keep
the highest `support` (no mean — we picked the highest-supporter in the
RANSAC winner, so this is well-defined).

## Evolution trail (kept for context)

| v   | step                              | result                         |
|-----|-----------------------------------|--------------------------------|
| v23 | 3-pt RANSAC, CPU                  | baseline                        |
| v24 | Ellipse Growing (Lin 2023 pattern)| N drops, fragile                |
| v25 | strict ratio constraint 1.10      | ratio filter                   |
| v26 | HoughLinesP arc-support (Lu 2020) | failed on thick contours        |
| v27 | 3C-FBI continuous voting         | edge-density biased             |
| v32 | skeleton-first architecture       | skeleton 50 ms becomes bottleneck |
| v35 | GPU-batched RANSAC inner          | 100 ms t_med (was 1-2 s)        |
| v36 | σ-driven adaptive arc_min         | no-op (σ range too narrow)      |
| v37 | reduced RANSAC budget (12×200)    | t −6 %, loses 2 turns / image    |
| v38 | support/r-driven adaptive arc_min | **gold standard hits GT=3**     |
| **v39** | v35 + v38 + 12×200 budget     | **production, 116 ms t_med**    |

Earlier versions are archived under `_archive/algo_v6/` and **are not committed**.

## Performance budget (100 images, RTX 4060 Ti)

| version | skeleton | RANSAC | refit + dedup | total t_med | <150 ms | <200 ms |
|---------|----------|--------|---------------|-------------|---------|---------|
| v35     | 51 ms    | 60 ms  | 13 ms         | 124 ms      | 69 %    | 92 %    |
| **v39** | 50 ms    | 53 ms  | 13 ms         | 116 ms      | **85 %**| **97 %**|

Speedup opportunities beyond v39:

1. **Write a custom CUDA kernel for the per-pixel `(d - r)² / (2·band²)`
   scoring + softmin reduction** — would replace `torch.cdist` + `torch.exp`
   with one fused kernel, projected −20 ms. Skeletonisation cannot be reduced
   further without losing connectivity.
2. **Move from arc_min filter to Bayesian model selection** — would replace
   the linear `support/r → arc_min` heuristic with a BIC-type score. Probably
   not worth it given gold-standard hit.
3. **Sub-pixel ellipse fit** (`MagHub`-style refinement) — would tighten
   axis ratio precision but does not affect N/t.
