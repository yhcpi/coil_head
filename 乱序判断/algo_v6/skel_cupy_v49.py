"""cupy Zhang-Suen 1984 + connected-component length-8 filter.

ZS84 alone produced 4.88× speedup but 50k spurious pixels/image (zigzag
artifacts along diagonal contours). Filter rule: only keep pixels that
belong to a connected component of size >= 8 — eliminates short zigzag
spurs while preserving long arcs that RANSAC needs.

This is the "long-branch preservation" idea from image thinning post-
processing (Fornaciari & Cucchiara, although their specifics are not
verified). Implementation is straightforward: scipy.ndimage.label +
bincount on labels.
"""
from __future__ import annotations
import numpy as np
import cupy as cp

_ZS_KERNEL = cp.RawKernel(r"""
extern "C" __global__ void zs_step(const unsigned char* P, const int H,
                                   const int W, const int sub,
                                   unsigned char* P_next, int* changed) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = H * W;
    if (idx >= total) return;
    const int y = idx / W;
    const int x = idx - y * W;
    if (y < 1 || y >= H-1 || x < 1 || x >= W-1) {
        P_next[idx] = P[idx];
        return;
    }
    const int P0 = P[idx];
    if (P0 == 0) { P_next[idx] = 0; return; }
    auto at = [&](int yy, int xx) -> int {
        if (yy < 0 || yy >= H || xx < 0 || xx >= W) return 0;
        return P[yy * W + xx];
    };
    int p2 = at(y-1, x);
    int p3 = at(y-1, x+1);
    int p4 = at(y, x+1);
    int p5 = at(y+1, x+1);
    int p6 = at(y+1, x);
    int p7 = at(y+1, x-1);
    int p8 = at(y, x-1);
    int p9 = at(y-1, x-1);
    int B = (p2==0 && p3==1) + (p3==0 && p4==1) + (p4==0 && p5==1) +
            (p5==0 && p6==1) + (p6==0 && p7==1) + (p7==0 && p8==1) +
            (p8==0 && p9==1) + (p9==0 && p2==1);
    int A = p2+p3+p4+p5+p6+p7+p8+p9;
    if (A < 2 || A > 6 || B != 1) { P_next[idx] = P0; return; }
    int cond;
    if (sub == 0) cond = (p2==0 || p6==0) && (p4==0 || p8==0);
    else cond = (p2==0 || p4==0) && (p6==0 || p8==0);
    if (cond) {
        P_next[idx] = 0;
        atomicAdd(changed, 1);
    } else {
        P_next[idx] = P0;
    }
}
""", "zs_step")


def zs_skeletonize(mask_bin: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Zhang-Suen thinning on GPU. Returns (x, y) int32 array of skeleton."""
    H, W = mask_bin.shape
    P = cp.asarray((mask_bin > 0).astype(np.uint8))
    P_next = cp.zeros_like(P)
    block = 256
    total = H * W
    grid = (total + block - 1) // block
    for _ in range(max_iter):
        changed = cp.zeros(1, dtype=cp.int32)
        _ZS_KERNEL((grid,), (block,),
                   (P, H, W, 0, P_next, changed))
        cp.cuda.runtime.deviceSynchronize()
        c0 = int(changed.get()[0])
        P, P_next = P_next, P
        if c0 == 0:
            break
        changed = cp.zeros(1, dtype=cp.int32)
        _ZS_KERNEL((grid,), (block,),
                   (P, H, W, 1, P_next, changed))
        cp.cuda.runtime.deviceSynchronize()
        c1 = int(changed.get()[0])
        P, P_next = P_next, P
        if c1 == 0:
            break
    ys, xs = cp.where(P > 0)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack([xs.get(), ys.get()]).astype(np.int32)


def filter_long_branches(skel_xy: np.ndarray, H: int, W: int,
                         min_size: int = 8) -> np.ndarray:
    """Keep only skeleton pixels in connected components of size >= min_size.

    RANSAC 3-pt circle needs ~30+ points on an arc to fit reliably.
    Short zigzag segments of size < 8 are too small to support a circle,
    so they are safely removed.
    """
    if len(skel_xy) == 0:
        return skel_xy
    skel_img = np.zeros((H, W), dtype=np.uint8)
    skel_img[skel_xy[:, 1].astype(int), skel_xy[:, 0].astype(int)] = 1
    from scipy.ndimage import label
    labels, n = label(skel_img, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return skel_xy[:0]
    sizes = np.bincount(labels.ravel())
    keep_labels = np.where(sizes >= min_size)[0]
    keep_labels = keep_labels[keep_labels > 0]
    if len(keep_labels) == 0:
        return skel_xy[:0]
    keep_mask = np.isin(labels, keep_labels)
    ys, xs = np.where(keep_mask)
    return np.column_stack([xs, ys]).astype(np.int32)
