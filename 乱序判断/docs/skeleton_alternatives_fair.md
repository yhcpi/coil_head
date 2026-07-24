# Skeleton alternatives — fair 4-way comparison (2026-07-24)

Background: v39 production uses `skimage.morphology.skeletonize(method='lee')`
which costs ~65 ms on 1440×1900 coil masks. The user asked whether a custom
cupy GPU Zhang-Suen kernel could push t_med below 50 ms without losing N.

We tested 4 alternatives on 100 images vs the v39 production baseline.

## Methods

| version | algorithm                            | code        |
|---------|--------------------------------------|-------------|
| v39     | skimage Lee 1994 (CPU)               | `v39.py`    |
| v40     | cupy ZS84 raw GPU kernel             | `v40.py`    |
| v41     | cupy ZS84 + GPU endpoint prune       | `v41.py`    |
| v42     | cv2.distanceTransform + 3×3 NMS      | `v42.py`    |
| v43     | cv2.ximgproc.thinning (Guo-Hall 1989) | `v43.py`    |

Each v4{0,1,2,3} uses v35 GPU 3-pt RANSAC + v38 support/r arc filter.
Only the *skeletonisation step* differs.

## Headline numbers (100 images, RTX 4060 Ti)

| version | t_med  | N_med  | gold 017_004 | verdict |
|---------|--------|--------|--------------|---------|
| **v39** | 108 ms | 4.3    | 2            | baseline |
| v40     | 102 ms | 8.1    | 7            | ✗ N doubled |
| v41     | 94 ms  | 7.8    | 4            | ✗ N doubled |
| v42     | **49 ms** | **0.0** | 1        | ✗ N broken |
| v43     | 121 ms | 4.7    | 4            | ≈ same |

`<150 ms` fraction:

| version | <150 ms | <200 ms |
|---------|---------|---------|
| v39     | 96 %    | 100 %   |
| v40     | 89 %    | 98 %    |
| v41     | 96 %    |        |
| v42     | 100 %   | 100 %   |

## Why each alternative fails

### v40 (cupy ZS84, 4.88× skel speed in isolation)
Zhang-Suen 1984 is known to produce **zigzag artifacts** (staircase pattern)
along diagonal contours. The cupy kernel is 4.88× faster than skimage Lee at
the *isolated* skel step (26.8 ms vs 130.7 ms), but the resulting skeleton has
~50k extra spurious pixels per image (vs ~15k Lee). These spurious short
branches cause 3-pt RANSAC to find many fake circles, **N_med 4 → 8**.

### v41 (cupy ZS84 + endpoint prune)
Added a GPU endpoint-prune kernel (`skel_cupy_prune.py`) that iteratively
deletes single-tips (1-neighbour active pixels) up to `min_branch` times.
mn=5: drops N from 8 to 7.8 (no real impact). Pruning helps if the spurious
pixels are *spike-shaped endpoint branches*, but ZS84's spurious pixels are
*interior zigzag* segments — each interior pixel has 2 neighbours so the
endpoint classifier doesn't tag them.

### v42 (cv2.distanceTransform + NMS)
DT is 10× faster than skimage Lee (20 ms vs 230 ms in isolation). The
3×3 non-max suppression produces only ~14 k peaks per image vs 35 k Lee
pixels. Those peaks are *scattered local maxima*, not connected line.
3-pt RANSAC needs many points along an arc to fit; with peaks sparse along
each ring, the support threshold (400 pixels) eliminates all hypotheses.

### v43 (cv2.ximgproc.thinning, Guo-Hall 1989)
Guo-Hall is 2.22× faster than skimage Lee in isolation (57 ms vs 128 ms).
In the integrated v39 pipeline the saving was only ~2 ms because skeleton is
no longer the bottleneck — RANSAC + dedup dominates. Essentially neutral.

## Time breakdown for v39 (where the cost really is)

| stage                  | t_med  | fraction |
|------------------------|--------|----------|
| skeleton (skimage Lee) | 65 ms  | 60 %     |
| RANSAC (12×200)        | 33 ms  | 30 %     |
| ellipse refit + dedup  | 10 ms  | 10 %     |
| **total**              | 108 ms |          |

The skel step is *not* the main cost. Even reducing skel to 5 ms (≥13×) saves
only 60 ms of the 108 ms total. This is why skeleton-only speedups underdeliver.

## Recommendation

**Keep v39 baseline (skimage Lee).** None of the four alternatives beats v39
on both N and t. v42 (DT) is the only one that *would* meet the user's t-med
target if N could be recovered, but N=0 on 98/100 images is a fatal regression.

If future speed gains are needed, the better attack surface is the 33-ms
**RANSAC inner loop**, not the 65-ms skeleton.

## Files

- `v40.py` `v41.py` `v42.py` `v43.py` — the 4 alternatives
- `skel_cupy.py` — cupy ZS84 raw kernel
- `skel_cupy_prune.py` — GPU endpoint prune
- `bench_v40_100.py` `bench_v42_100.py` `sweep_v41_mb.py` — benchmarks
- `v4{0,1,2}_bench.json` — raw 100-image stats
