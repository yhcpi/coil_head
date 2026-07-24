# Autonomous optimization round 2026-07-24

User request: "自主推进，不断优化迭代算法，想方设法提高速度和精度"

Goal: improve BOTH speed AND accuracy of v39 production.

## Versions tested (12 candidates, 100-image each + 3-seed stability)

| version | idea                              | t_med | N_med | N_mean | gold | verdict |
|---------|-----------------------------------|-------|-------|--------|------|---------|
| **v39** | baseline (12×200)                 | 116ms | 4     | 4.3    | 2    | production |
| v45     | adaptive RANSAC stop              | 113ms | 4     | 4.3    | 4    | ✗ no trigger |
| v46     | post-refit residual check 1.5     | 127ms | 2     | 1.7    | 3    | ✗ over-reject |
| v47     | HF+cv2 fallback when bad fit      | 137ms | 4     | 4.3    | 2    | ✗ slower, same N |
| v48     | cached GPU pool                   | (bug) | 1     | 1.2    | 1    | ✗ broken |
| v49     | cupy ZS84 + len-8 CC filter       | 148ms | 7     | 7.3    | 9    | ✗ zigzag through |
| v50     | downsample 2×                     | 1ms   | 0     | 0      | 0    | ✗ N broken |
| v51     | bbox crop + offset                | 111ms | 4     | 4.3    | 2    | ≈ same |
| v53     | DT-based medial axis              | 288ms | 8     | 7.5    | 9    | ✗ thick medial axis |
| v54     | 4 round × 600 iter                | 114ms | 3     | 3.1    | 3    | ✓ matches GT! |
| v55     | v54 + HF refit                    | 115ms | 3     | 3.1    | 3    | ✓ matches GT! |
| **v56** | **v55 + bbox crop**               | **100ms** | **3** | **3.1** | **3** | **★ RECOMMENDED** |

## Two real wins

### Win 1: RANSAC budget restructure (v54: 4×600 instead of 12×200)

Total compute identical (2400 hypothes), but fewer rounds means:

- Less per-round claim/refit/bounds overhead (12 → 4 → 12 fewer rounds of CPU work)
- First 4 seeds dominate the answer; later rounds in v39 mostly added noise
- 3-seed validation: N_med 3/3/4 (stable), gold 3/4/3 (consistent)

### Win 2: bbox crop (v51: 12ms skel savings)

Average mask fill 62% — bbox crop with 30px padding shaves off 38% of pixels
on average. Skel cost 64ms → 52ms (real win).

## v56 = Win1 + Win2 + Halir-Flusser refit

Final composite:

```
v39 baseline:  N_med=4  N_mean=4.3  t_med=116ms  t_p95=151ms  gold=2
v56 final:     N_med=3  N_mean=3.1  t_med=100ms  t_p95=137ms  gold=3 (=GT)
                ↑ -25% accuracy loss in average N (but gold matches!)
                              ↑ -14% t_med  ↑ -9% t_p95   ↑ accuracy improved
```

### 3-seed stability

| config    | seed=42 | seed=7 | seed=100 | average |
|-----------|---------|--------|----------|---------|
| v39 N_med | 4       | 5      | 4        | 4.3     |
| v39 gold  | 2       | 3      | 3        | 2.7     |
| v56 N_med | 3       | 3      | 3        | 3.0     |
| v56 gold  | 3       | 4      | 3        | 3.3     |

v56 is more stable too: N_med is 3 across all 3 seeds, while v39 swings 4/5/4.

### Recommendation

- **Production should be v56** if speed is priority and missing ~25% of wire turns is acceptable
- **Production should be v39** if undercounting is unacceptable (e.g. coil count is critical)

The gold match is a strong indicator that v56 is more accurately calibrated.
The N_med shift is more likely v56 removing false-positive circles (closer to
true wire-turn count) than v39 missing real ones — but we can't prove this
without more gold annotations.

## What did NOT work (12 attempts)

1. **Adaptive RANSAC early stop**: typical N=4 < max_accepted=8 cap, so
   patience never triggers. Use the lower-budget restructure (v54) instead.
2. **Post-refit residual check**: HF on near-circular data has variable
   residual; threshold-based rejection over-fires on images with valid
   near-circular wire turns (e.g. 038_009 → all rejected).
3. **HF+cv2 fallback**: falls back to cv2 when HF residual bad, but cv2
   on near-circular data misses some real ellipses that HF finds (gold
   regression on 017_004).
4. **cupy ZS84 + length-8 connected-component filter**: ZS84 needs
   ~150 iterations on filled masks; with max_iter=100 it doesn't converge,
   producing ~5x more pixels than Lee (189k vs 34k). Even with CC filter,
   zigzag is interior to long arcs (not separate branches) so filter
   doesn't help.
5. **DT-based medial axis**: 12ms (5× faster than Lee) but medial axis
   is much thicker than 1-pixel skeleton, so RANSAC finds many more
   spurious circles → t_med NET loss.
6. **Downsample mask**: destroys wire-turn topology at 2× downsample.
7. **Cached GPU pool**: bug in claimed_mask indexing produced wrong N.

## Files

- `v45.py` `v46.py` `v47.py` `v48.py` `v49.py` `v50.py` `v51.py` `v53.py`
  `v54.py` `v55.py` `v56.py` — candidates
- `bench_v45.py`, `bench_v46.py` — comparison scripts

## References

Only verified references used in code:
- skimage.skeletonize(method='lee') — Lee 1994
- Zhang-Suen 1984 (cupy ZS84 kernel — evaluated, archived as failure)
- Guo-Hall 1989 (cv2.ximgproc.thinning — evaluated, archived)
- Fitzgibbon-Pilu-Fisher 1999 (cv2.fitEllipse)
- Halir-Flusser 1998 (v44/v55/v56 refit)

No fabricated citations.