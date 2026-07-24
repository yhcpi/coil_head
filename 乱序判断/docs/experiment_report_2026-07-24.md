# 3-direction experiment report (2026-07-24)

Following the user-stated plan "按照以下计划，仔细推进实验":
1. MagHub sub-pixel refinement → 提升 a/b ±1 pixel
2. PIE-NET 微调 mask input → one-shot N estimation
3. 自写 cupy skeleton → t_med −30ms

## 1. MagHub / Halir-Flusser sub-pixel ellipse refit (v44)

**Outcome**: implemented Halir-Flusser 1998 from verified literature. Sub-pixel
precision improvement on synthetic data (a err 5.0 → 0.006 px). On real 100
images, equivalent to v39 (N_med 5 vs 4, gold 4 vs 2 vs GT=3, t_med 122 vs
118 ms). See `refit_backend_choice.md`.

**Note**: "MagHub" turned out to be unverified. The actual implementation
is Halir-Flusser 1998 (Halir, Flusser. "Numerically stable direct least
squares fitting of ellipses". 6th WSCG, 1998) — verified by WebSearch
returning real DOI / Python reference (bdhammel/least-squares-ellipse-fitting).

## 2. PIE-NET one-shot N estimation

**Outcome**: **skipped** — paper arxiv ID 2205.08059 unverified by
WebSearch. No usable source code or paper PDF could be located. Building a
6-8 h training pipeline against an unverifiable method is not justified.

The v33 EM-hard-assignment (in production) handles dedup after RANSAC; a
CNN-based N estimator would only be warranted if RANSAC itself were the
bottleneck. As of v39, RANSAC is 30 % of t_med (33/108 ms) — too small to
justify the training cost. Skipped by design.

## 3. Custom cupy GPU Zhang-Suen skeleton (v40/v41/v42/v43)

**Outcome**: tested 4 alternatives vs v39's skimage Lee baseline.
See `skeleton_alternatives_fair.md` for full numbers.

| version | algorithm | t_med | N_med | verdict |
|---------|-----------|-------|-------|---------|
| v39     | skimage Lee 1994 | 108 ms | 4.3 | baseline |
| v40     | cupy ZS84 GPU    | 102 ms | 8.1 | ✗ N doubled |
| v41     | cupy ZS84 + prune| 94 ms  | 7.8 | ✗ N doubled |
| v42     | DT + NMS         | 49 ms  | 0.0 | ✗ N broken |
| v43     | Guo-Hall ximgproc| 121 ms | 4.7 | ≈ neutral |

cupy ZS84 is 4.88× faster than skimage Lee at the isolated skel step
(26.8 vs 130.7 ms), but the resulting skeleton has ~50k spurious pixels per
image that produce fake RANSAC circles. Endpoint-pruning (v41) doesn't help
because ZS84's spurious pixels are *interior zigzag* segments, not
endpoint branches.

## Synthesis

**Production stays at v39.** None of the three directions delivers a clear
N or t improvement over the v39 baseline. The path forward, if any, is the
33-ms RANSAC inner loop (30 % of t_med), not the 65-ms skeleton (60 % of
t_med). And sub-pixel ellipse refit alone does not help N because RANSAC
dominates N decisions.

**Citation hygiene**: per user feedback ("已经用到我的代码算法的，正是有用
的，我才读，没用过的我不读"), only verified references are cited:
- Lee 1994 (skimage.skeletonize)
- Zhang-Suen 1984 (cupy kernel — but the kernel is archived as failure)
- Guo-Hall 1989 (cv2.ximgproc.thinning)
- Fitzgibbon-Pilu-Fisher 1999 (cv2.fitEllipse)
- Halir-Flusser 1998 (v44)

The originally proposed "MagHub" and "PIE-NET 2205.08059" were unverifiable
and have been removed from consideration.

## Files

- `v40.py`, `v41.py`, `v42.py`, `v43.py` — skeleton alternatives (archived as
  failures)
- `v44.py`, `ellipse_halir_flusser.py` — refit alternative (equivalent to v39)
- `bench_v44_100.py`, `bench_v40_100.py`, `bench_v42_100.py` — benchmarks
- `skel_cupy.py`, `skel_cupy_prune.py` — failed cupy kernels
- `v4{0,1,2,4}_bench.json` — raw stats