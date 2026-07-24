"""GPU Zhang-Suen thinning via CuPy.

This is the Zhang-Suen 1984 algorithm ported to a parallel GPU kernel.
Each iteration has 2 subiterations; in each subiter, every pixel can be
flipped from 1 to 0 in parallel based on its 8-neighbours.

Algorithm references:
- Zhang, T.Y. & Suen, C.Y. (1984) "A fast thinning algorithm for thinning
  digital patterns", CACM 27(3): 236-239. [no DOI; widely cited]
- Non-parallel canonical implementation: skimage.morphology.skeletonize
  (Lee 1994 fast method, differs from ZS but ends at 1-pixel skeleton)

Important:
- We must GPU-synchronize after each subiter (rows depend on cols, both depend
  on previous state)
- We add `changed.sum() == 0` exit condition to stop when converged
- Padded arrays with npad=1 to avoid boundary checks inside kernel
"""
from __future__ import annotations
import time
from typing import Any

import cv2
import numpy as np
import cupy as cp


_PADDED_FROM_BLOCK = cp.RawKernel(
    """
    extern "C" __global__
    void pad_ones(const int H, const int W,
                  const int* __restrict__ mask,
                  const int npad,
                  int* __restrict__ padded) {
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        const int Hp = H + 2 * npad;
        const int Wp = W + 2 * npad;
        if (i >= Hp || j >= Wp) return;
        const int pi = i - npad;
        const int pj = j - npad;
        int v = 1;
        if (pi >= 0 && pi < H && pj >= 0 && pj < W) {
            v = mask[pi * W + pj];
        }
        padded[i * Wp + j] = v;
    }
    """,
    "pad_ones",
)


_ZS_SUBITER = cp.RawKernel(
    """
    /* Zhang-Suen subiteration.
       p2 p3 p4   (clockwise from north, starting at top)
       p1 P  p5
       p8 p7 p6

       P is current pixel (1 = active).
       sub = 0: deletion = north/south (P2=0 | P6=0) & (P4=0 & P8=0)
       sub = 1: deletion = east/west   (P2=0 | P4=0) & (P6=0 & P8=0)
    */
    extern "C" __global__
    void zs_step(const int H, const int W,
                 const int* __restrict__ P,
                 int* __restrict__ P_next,
                 const int sub,
                 int* __restrict__ changed) {
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= H || j >= W) return;
        const int Wp = W + 2;
        const int idx = (i + 1) * Wp + (j + 1);
        const int P0 = P[idx];
        if (P0 == 0) { P_next[idx] = 0; return; }
        // neighbours, p2..p9 (0..7) cycling clockwise starting north
        const int p2 = P[idx - Wp];
        const int p3 = P[idx - Wp + 1];
        const int p4 = P[idx + 1];
        const int p5 = P[idx + Wp + 1];
        const int p6 = P[idx + Wp];
        const int p7 = P[idx + Wp - 1];
        const int p8 = P[idx - 1];
        const int p9 = P[idx - Wp - 1];
        const int B = (p2 == 0 && p3 == 1 ? 1 : 0)
                    + (p3 == 0 && p4 == 1 ? 1 : 0)
                    + (p4 == 0 && p5 == 1 ? 1 : 0)
                    + (p5 == 0 && p6 == 1 ? 1 : 0)
                    + (p6 == 0 && p7 == 1 ? 1 : 0)
                    + (p7 == 0 && p8 == 1 ? 1 : 0)
                    + (p8 == 0 && p9 == 1 ? 1 : 0)
                    + (p9 == 0 && p2 == 1 ? 1 : 0);
        const int A = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
        if (A < 2 || A > 6 || B != 1) { P_next[idx] = P0; return; }
        int cond;
        if (sub == 0) {
            cond = ((p2 == 0 || p6 == 0) && (p4 == 0 || p8 == 0));
        } else {
            cond = ((p2 == 0 || p4 == 0) && (p6 == 0 || p8 == 0));
        }
        P_next[idx] = (cond ? 0 : P0);
        if (cond) atomicAdd(changed, 1);
    }
    """,
    "zs_step",
)


def zhang_suen_cupy(mask_bin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zhang-Suen thinning on GPU. Returns (skel_pixels_xy, skel_image)."""
    H, W = mask_bin.shape[:2]
    # Send mask to GPU as 0/1 ints, padded with 1 to avoid boundary artefacts
    mask_i32 = cp.asarray(mask_bin.astype(np.int32))
    Hp = H + 2
    Wp = W + 2
    P = cp.zeros((Hp, Wp), dtype=cp.int32)
    # Use a basic kernel to pad. For simplicity, use cp.pad (one-time cost):
    P[1:H+1, 1:W+1] = mask_i32
    P[0, :] = 1; P[-1, :] = 1; P[:, 0] = 1; P[:, -1] = 1
    block = (32, 32)
    grid = ((Wp + block[0] - 1) // block[0], (Hp + block[1] - 1) // block[1])
    changed = cp.zeros(1, dtype=cp.int32)
    P_next = cp.zeros_like(P)
    max_iter = 300
    for it in range(max_iter):
        for sub in (0, 1):
            changed[0] = 0
            _ZS_SUBITER(grid, block, (Hp, Wp, P, P_next, sub, changed))
            P, P_next = P_next, P
        cp.cuda.Stream.null.synchronize()
        if int(changed[0]) == 0:
            break
    skel = P[1:H+1, 1:W+1].get()
    ys, xs = np.where(skel > 0)
    skel_pix = np.column_stack((xs, ys)).astype(np.int32)
    return skel_pix, skel.astype(np.uint8)


# ---- benchmark ----

def bench(mask_path: str, *, n_iter: int = 10):
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_bin = (img > 127).astype(np.uint8)
    # Warm-up
    _ = zhang_suen_cupy(mask_bin)
    cp.cuda.Stream.null.synchronize()
    times = []
    for _ in range(n_iter):
        t0 = time.time()
        _ = zhang_suen_cupy(mask_bin)
        cp.cuda.Stream.null.synchronize()
        times.append((time.time() - t0) * 1000)
    times = np.array(times)
    return {
        "t_med_ms": float(np.median(times)),
        "t_min_ms": float(np.min(times)),
        "t_mean_ms": float(np.mean(times)),
        "n": n_iter,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    masks_dir = here.parent / "yhc" / "mask_refine"
    files = sorted(masks_dir.glob("*.png"))
    print(f"GPU ZS vs skimage on 30 sample masks...")
    sample = files[:30] if len(files) > 30 else files
    # 1. skimage timing
    from skimage.morphology import skeletonize
    sk_times = []
    cupy_times = []
    pixel_diffs = []
    for f in sample:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        mask_bin = (img > 127).astype(np.uint8)
        # skimage warm
        _ = skeletonize(mask_bin, method='lee')
        # skimage run
        t0 = time.time()
        sk = skeletonize(mask_bin, method='lee').astype(np.uint8)
        sk_times.append((time.time() - t0) * 1000)
        # cupy warm
        _ = zhang_suen_cupy(mask_bin)
        t0 = time.time()
        cp_pix, cp_skel = zhang_suen_cupy(mask_bin)
        cp.cuda.Stream.null.synchronize()
        cupy_times.append((time.time() - t0) * 1000)
        # pixel diff
        diff = int(np.sum(sk != cp_skel))
        pixel_diffs.append(diff)
    sk_times = np.array(sk_times)
    cupy_times = np.array(cupy_times)
    pixel_diffs = np.array(pixel_diffs)
    print(f"\n========== skeleton benchmark (30 images) ==========")
    print(f"skimage (Lee):  t_med {np.median(sk_times):.1f}ms  "
          f"t_min {np.min(sk_times):.1f}ms  t_mean {np.mean(sk_times):.1f}ms")
    print(f"cupy (ZS84):    t_med {np.median(cupy_times):.1f}ms  "
          f"t_min {np.min(cupy_times):.1f}ms  t_mean {np.mean(cupy_times):.1f}ms")
    print(f"speedup vs skimage: med {np.median(sk_times)/np.median(cupy_times):.2f}x, "
          f"min {np.min(sk_times)/np.min(cupy_times):.2f}x")
    print(f"\npixel diff vs skimage: med {np.median(pixel_diffs)}  "
          f"mean {pixel_diffs.mean():.0f}  max {pixel_diffs.max()}")
    print(f"# exact match: {(pixel_diffs == 0).sum()}/{len(pixel_diffs)}")
