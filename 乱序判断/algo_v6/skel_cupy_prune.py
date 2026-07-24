"""GPU skeleton pruning: remove short ZS84 spurious branches.

After Zhang-Suen 1984, the skeleton has many short end-branches (length 1-5px)
that arise from jagged diagonal contours. They cause spurious RANSAC detections.

Algorithm (Liu 2014 / Saeed 2010 — standard morphological pruning on thinned image):
  A skeleton pixel is an "endpoint" if it has exactly 1 active 8-neighbour.
  Recursively delete endpoints until each remaining branch has length >= min_branch.
  Repeats until stability.

Simpler & GPU-friendly: iterative endpoint deletion with a length budget:
  Step 1: classify each active pixel by #(active 8-neighbours) ∈ {1: end, 2: pass, 3+: junction}
  Step 2: mark all endpoints for deletion, recompute, until no further endpoints have
          branch-length < min_branch on either side.

For GPU parallelism we use the *single-iteration pruning* per pixel:
  - A pixel is deleted if it is an endpoint AND its connected branch length
    to either the nearest junction or the boundary is < min_branch.
  - This is approximated by counting the number of deletion-eligible steps
    before the chain reaches a junction.

Implementation:
  - We run deletion iteratively: at each iter, mark endpoints whose
    deletion would not cascade a branch (i.e., they are "true ends" of
    deletable branches).
  - After K iters, only paths of length ≥ K remain.
  - We stop when K == min_branch.
"""
from __future__ import annotations
import time
import numpy as np
import cv2
import cupy as cp


# Kernel: classify each active pixel by neighbour count, mark endpoints (==1).
_CLASSIFY = cp.RawKernel(
    """
    extern "C" __global__
    void classify(const int H, const int W,
                  const int* __restrict__ P,
                  int* __restrict__ is_end) {
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= H || j >= W) return;
        const int idx = i * W + j;
        const int P0 = P[idx];
        if (P0 == 0) { is_end[idx] = 0; return; }
        const int p1 = (i > 0     && j > 0)     ? P[(i-1)*W + j-1] : 0;
        const int p2 = (i > 0)                  ? P[(i-1)*W + j]   : 0;
        const int p3 = (i > 0     && j+1 < W)   ? P[(i-1)*W + j+1] : 0;
        const int p4 = (j+1 < W)                ? P[i*W + j+1]     : 0;
        const int p5 = (i+1 < H && j+1 < W)     ? P[(i+1)*W + j+1] : 0;
        const int p6 = (i+1 < H)                ? P[(i+1)*W + j]   : 0;
        const int p7 = (i+1 < H && j > 0)       ? P[(i+1)*W + j-1] : 0;
        const int p8 = (j > 0)                  ? P[i*W + j-1]     : 0;
        const int A = p1+p2+p3+p4+p5+p6+p7+p8;
        is_end[idx] = (A == 1) ? 1 : 0;
    }
    """,
    "classify",
)


# Kernel: delete all endpoints at once (atomic counter)
_DELETE_ENDPOINTS = cp.RawKernel(
    """
    extern "C" __global__
    void delete_ends(const int H, const int W,
                     const int* __restrict__ P,
                     const int* __restrict__ is_end,
                     int* __restrict__ P_next,
                     int* __restrict__ changed) {
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= H || j >= W) return;
        const int idx = i * W + j;
        if (is_end[idx]) {
            P_next[idx] = 0;
            atomicAdd(changed, 1);
        } else {
            P_next[idx] = P[idx];
        }
    }
    """,
    "delete_ends",
)


def prune_short_branches(skel_pix_xy: np.ndarray, mask_shape: tuple,
                         min_branch: int = 5) -> np.ndarray:
    """Remove ZS spurious short branches.

    Args:
      skel_pix_xy: list of (x, y) skeleton pixel coords.
      mask_shape: (H, W).
      min_branch: branches shorter than this get removed.

    Returns: filtered (x, y) np.ndarray.
    """
    H, W = mask_shape
    P = cp.zeros((H, W), dtype=cp.int32)
    if len(skel_pix_xy) == 0:
        return skel_pix_xy
    xs = cp.asarray(skel_pix_xy[:, 0].astype(np.int32))
    ys = cp.asarray(skel_pix_xy[:, 1].astype(np.int32))
    P[ys, xs] = 1
    block = (32, 32)
    grid = ((W + block[0] - 1) // block[0], (H + block[1] - 1) // block[1])
    is_end = cp.zeros((H, W), dtype=cp.int32)
    P_next = cp.zeros_like(P)
    changed = cp.zeros(1, dtype=cp.int32)
    # Iteratively delete endpoints up to min_branch times.
    # We stop if a step deletes nothing.
    for it in range(min_branch):
        _CLASSIFY(grid, block, (H, W, P, is_end))
        cp.cuda.Stream.null.synchronize()
        # check if any endpoint
        n_ends = int(is_end.sum())
        if n_ends == 0:
            break
        changed[0] = 0
        _DELETE_ENDPOINTS(grid, block, (H, W, P, is_end, P_next, changed))
        P, P_next = P_next, P
        cp.cuda.Stream.null.synchronize()
        if int(changed[0]) == 0:
            break
    ys_out, xs_out = cp.where(P > 0)
    xs_out = xs_out.get(); ys_out = ys_out.get()
    return np.column_stack((xs_out, ys_out)).astype(np.int32)


def bench_prune(skel_pix_list, mask_shape, min_branch_list=(3, 5, 8, 12)):
    import json
    out = {}
    for mb in min_branch_list:
        # warm
        _ = prune_short_branches(skel_pix_list[0], mask_shape, min_branch=mb)
        cp.cuda.Stream.null.synchronize()
        times = []
        out_counts = []
        for sk in skel_pix_list:
            t0 = time.time()
            out_pix = prune_short_branches(sk, mask_shape, min_branch=mb)
            cp.cuda.Stream.null.synchronize()
            times.append((time.time() - t0) * 1000)
            out_counts.append(len(out_pix))
        times = np.array(times)
        out_counts = np.array(out_counts)
        out[f"mb={mb}"] = {
            "t_med": float(np.median(times)),
            "n_med": float(np.median(out_counts)),
            "n_mean": float(out_counts.mean()),
        }
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    project = here.parent
    masks_dir = project / "yhc" / "mask_refine"
    files = sorted(masks_dir.glob("*.png"))
    sample = files[:30] if len(files) > 30 else files
    import skel_cupy
    skel_pixels_list = []
    shapes = []
    print(f"ZS-thinning {len(sample)} images for pruning test...")
    for f in sample:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        mask_bin = (img > 127).astype(np.uint8)
        sp, _ = skel_cupy.zhang_suen_cupy(mask_bin)
        skel_pixels_list.append(sp)
        shapes.append(mask_bin.shape)
    out_n = np.array([len(s) for s in skel_pixels_list])
    print(f"ZS pixels: med {np.median(out_n):.0f} mean {out_n.mean():.0f} max {out_n.max()}")
    print(f"\n========== GPU prune sweep (30 images) ==========")
    for k, v in bench_prune(skel_pixels_list, shapes[0]).items():
        print(f"{k}: t_med={v['t_med']:.1f}ms  N_med={v['n_med']:.0f}  "
              f"N_mean={v['n_mean']:.0f}")
