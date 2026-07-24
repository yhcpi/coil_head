# 钢卷乱序检测 / Coil Order Detection

Wire-turn elliptical detection in 2D binary masks of steel coil end-surfaces.

## What this does

For each binary mask in `yhc/mask_refine/`, the algorithm estimates
**the number of visible wire turns** (≈ real concentric ellipses on the coil
end-face) and the **radius / eccentricity / orientation** of each.

Pipeline:

```
binary mask  →  skeleton  →  3-pt circle RANSAC (GPU)
           →  support/r-adaptive arc filter  →  ellipse refit
           →  r-cluster dedup  →  wire-turn list (cx, cy, a, b, ang)
```

Final production version: **`algo_v6/v39.py`**.

## Quick start

```bash
cd algo_v6
# run on gold-standard image 017_roi_004 (GT = 3 wire turns)
python v39.py 017_roi_004
# → renders canonical/v39_017_roi_004.png with detected ellipses

# full 100-image benchmark
python bench_v39_100.py
```

## Results (2026-07-24, 100-image evaluation)

| version | description             | N_med | N_mean | t_med | <150ms |
|---------|-------------------------|-------|--------|-------|--------|
| v35     | GPU batched RANSAC base | 7     | 7.3    | 124ms | 69%    |
| v38     | + support/r arc filter  | 5     | 4.8    | 127ms | 71%    |
| **v39** | + n_iter=200             | **4** | **4.3**| **116ms** | **85%** |

**Gold standard `017_roi_004` (GT = 3 wire turns):**

| v35 | v38 | v39 |
|-----|-----|-----|
| 6-9 (overcount) | **3 ✓** | 2 (undercount by 1) |

95/100 images: v38 / v39 detect **fewer** circles than v35 (precision ↑).
Gold standard confirms those were spurious overcounts.

## Layout

```
乱序判断/
├── README.md                ← this file
├── algo_v6/                 ← production algorithm
│   ├── v32.py               ← skeleton-first base (skimage.skeletonize Lee94)
│   ├── v35.py               ← GPU-batch 3-pt RANSAC inner loop
│   ├── v38.py               ← adaptive arc_min via support/r ratio
│   ├── v39.py               ← production (v35 + v38 + n_iter=200)
│   ├── skel_torch.py        ← GPU Zhang-Suen reference impl (slower, kept for reference)
│   ├── bench_param_sweep.py ← RANSAC budget sweep
│   ├── bench_v38_100.py
│   ├── bench_v39_100.py
│   ├── gt_ellipses.json     ← 017_roi_004 gold standard
│   └── canonical/           ← v35 / v38 / v39 renderings (3 images × 3 versions)
├── docs/
│   ├── algorithm.md         ← chain + key insights
│   ├── benchmark.md         ← v35 / v38 / v39 numbers + gold standard
│   └── figures/             ← sample panel inputs + outputs
├── yhc/
│   └── mask_refine/         ← 100 final binary masks (100 PNGs, 4.3 MB)
├── 017_roi_004.png          ← gold standard raw coil image
├── 038_roi_001_mask.png     ← sample mask overlay
├── 038_roi_001_raw.png      ← sample raw coil image
├── 乱序.txt                   ← original problem statement
├── 线圈.pdf                   ← reference paper
└── _archive/                ← pre-v39 experimental files (not committed)
```

## Algorithm at a glance (full detail in `docs/algorithm.md`)

1. **Skeleton** (`v32._skeleton`): `skimage.morphology.skeletonize` (Lee 1994, ~50 ms)
2. **3-pt-circle RANSAC** (`v35._ransac_all_rounds_gpu`): GPU-batched 3-pt circle algebra
   + adaptive band scoring. 12 rounds × 200 iter ≈ 100 ms on RTX 4060 Ti.
3. **Adaptive arc filter** (`v38.adaptive_arc_min_support`): trust circles with
   high support/r ratio (more perimeter pixels per radius → real wire turn);
   reject sparse candidates (likely partial-arc false positive).
4. **Ellipse refit** (`v32._fit_ellipse_safe`): cv2.fitEllipse on inliers with
   band ±k·σ MAD-residual radius.
5. **r-cluster dedup** (`v32._r_cluster_dedup`): keeps highest-score circle in
   each radius cluster (within dr_tol_px).

## References

- L. Lam, S.-W. Lee, C. Y. Suen, *Thinning Methodologies — A Comprehensive Survey*, IEEE TPAMI 1992.
- T.-C. Lee, *Building Skeleton Models via 3-D Medial Surface / Thinning Algorithms*, CVGIP 1994.
- A. Fitzgibbon, M. Pilu, R. Fisher, *Direct Least-Squares Fitting of Ellipses*, IEEE TPAMI 1999 (DOI 10.1109/34.765658).
- R. Schnabel, R. Klein, *Octree-based Fusion of RANSAC and Least-Squares Estimation*, 2007.
- D. Barath, J. Matas, *MAGSAC / MAGSAC++: Marginalizing Sample Consensus*, CVPR 2019 / 2020.

See `docs/algorithm.md` for in-text citations and full algorithm walk-through.

## Provenance

Generated during the *乱序判断* project (coil-order detection). Author: yhcpi.
Trajectory v23→v39 captured in `docs/algorithm.md` § "Evolution".
