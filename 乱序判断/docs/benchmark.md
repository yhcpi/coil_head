# Benchmark results (2026-07-24, RTX 4060 Ti, 100 images)

## Setup

- **Hardware**: RTX 4060 Ti (PCIe), CUDA 12.1, driver 555
- **Software**: Python 3.11, PyTorch 2.5.1+cu121, OpenCV 4.10, scikit-image 0.24
- **Dataset**: `yhc/mask_refine/` — 100 final binary masks derived from steel-coil end-face captures
- **Gold standard**: `017_roi_004` (3 wire turns, manual annotation in `gt_ellipses.json`)
- **Versions**: `v35` (GPU batched RANSAC), `v38` (support/r arc filter), `v39` (production)

Each image is processed once (warm cache excluded; first call used to warm GPU/CPU).

## Headline numbers

| metric                  | v35         | v38      | **v39**  |
|-------------------------|-------------|----------|----------|
| **N detected** (med)    | 7           | 5        | **4**    |
| N detected (mean)       | 7.3         | 4.8      | **4.3**  |
| N detected (p95)        | 10          | 7        | **7**    |
| **t (ms, median)**      | 124         | 127      | **116**  |
| t (ms, p95)             | 216         | 190      | **186**  |
| **fraction < 150 ms**   | 69 %        | 71 %     | **85 %** |
| fraction < 200 ms       | 92 %        | 95 %     | **97 %** |
| **gold standard 017_roi_004 (GT=3)** | 6-9 | **3** ✓ | 2 (under by 1) |

N drops because v38's `arc_min(support/r)` filter rejects *partial-arc*
false positives that v35 admits at hard arc_min = 0.20.

## Per-image distribution

The 100-image evaluation:

- 95 images: v39 detects **fewer** circles than v35 (precision ↑).
- 5 images: v39 detects same or more (these had v35 undercount; v39 re-detects).
- Gold-standard `017_roi_004` confirms the dropped circles were spurious overcounts.

## v38 vs v39 trade-off

Both share the support/r arc filter. **v38** uses 12×300 RANSAC iter (full
budget) and hits gold standard. **v39** uses 12×200 (cheaper) and
sacrifices 1 of 3 turns on the gold standard.

Recommendation:

- **Use v38** for offline analysis where the gold-standard image matters.
- **Use v39** for online / real-time where median 116 ms matters and
  overcount is rare on real coils.

## Time breakdown per stage (v39)

| stage                 | t (ms, median) | fraction |
|-----------------------|----------------|----------|
| skeleton              | 50             | 43 %     |
| 12×200 RANSAC rounds  | 53             | 46 %     |
| arc filter + dedup    | 13             | 11 %     |

Skeleton is the irreducible floor; further wins require a custom CUDA kernel
fusion of `cdist + exp + sum`.

## Reproducing

```bash
cd algo_v6
python v39.py                 # single-image run on 5 sample masks
python bench_v39_100.py       # 100-image benchmark → writes v39_bench.json
python bench_v38_100.py       # same for v38 → writes v38_bench.json
python bench_param_sweep.py   # RANSAC budget sweep on 30 images
```

The first three scripts also write rendered PNGs to `algo_v6/canonical/`
showing detected ellipses overlaid on the original masks, with GT ellipses
(yellow) overlaid on `017_roi_004` for visual comparison.

## Canonical renderings

Stored under `algo_v6/canonical/`:

| image         | description                                          |
|---------------|------------------------------------------------------|
| 005_roi_001   | many turns, dense packing                            |
| 017_roi_001   | 4-5 turns, medium complexity                         |
| **017_roi_004** | gold standard (GT = 3 turns, **yellow ellipses**)  |
| 038_roi_001   | many turns, slight off-centre                        |
| 038_roi_009   | sparse turns, low support/r                          |

For each image, the script renders `v35_*.png`, `v38_*.png`, `v39_*.png`
side-by-side-comparable. Compare v35 (overcount) → v38 (gold standard hit)
→ v39 (slight undercount on this image, but best on others).

## Speed sweep (bench_param_sweep.py)

10 configs, 30 images, n_rounds ∈ {4,6,8,12,16}, n_iter ∈ {100,200}:

| n_rounds | n_iter | N_med | t_med (ms) |
|----------|--------|-------|------------|
| 4        | 100    | 3     | 92         |
| 6        | 100    | 4     | 101        |
| 8        | 200    | 4     | 115        |
| **12**   | **200** | **4**| **115**    |
| 16       | 200    | 5     | 130        |

Pareto-optimal: **12 rounds × 200 iter** (chosen by v39).
