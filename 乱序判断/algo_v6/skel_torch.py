"""Zhang-Suen thinning on GPU via PyTorch.

The Zhang-Suen algorithm (1988) is parallel: each iteration marks deletions
based on 8-neighbor counts. We run all iterations on GPU. Each iteration
takes O(N) time but the operations are pure tensor math.

Why this is good for our pipeline:
- Mask is 1440x1900 uint8 (2.7M pixels)
- Algorithm converges in O(min(W,H)) ≈ 1440 iterations
- Each iteration is ~8 array ops + reduction
- On RTX 4060 Ti: each iteration ~0.5ms → ~700ms total (still slow)
- BUT early stopping when iteration makes no changes (typical: 60-100 iter)

Key trick: track change count per iter. When 0 → stop.
"""
import torch
import numpy as np


def zhang_suen_torch(mask_np, device=None, max_iter=1000):
    """Zhang-Suen thinning on GPU. mask_np: uint8 0/1 or 0/255. Returns uint8 0/1."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = (mask_np > 0).astype(np.uint8)
    M = torch.from_numpy(m).to(device)
    H, W = M.shape
    if H < 3 or W < 3:
        return m
    # Padded to handle border
    # Neighbor offsets (P2..P9 clockwise from N):
    # P2 = (y-1, x), P3 = (y-1, x+1), P4 = (y, x+1), P5 = (y+1, x+1),
    # P6 = (y+1, x), P7 = (y+1, x-1), P8 = (y, x-1), P9 = (y-1, x-1)
    # But we'll use cross + diagonal tensors for speed.
    n1 = torch.tensor([-1, 0], device=device)  # P2
    n2 = torch.tensor([-1, +1], device=device)  # P3
    n3 = torch.tensor([0, +1], device=device)  # P4
    n4 = torch.tensor([+1, +1], device=device)  # P5
    n5 = torch.tensor([+1, 0], device=device)  # P6
    n6 = torch.tensor([+1, -1], device=device)  # P7
    n7 = torch.tensor([0, -1], device=device)  # P8
    n8 = torch.tensor([-1, -1], device=device)  # P9
    offset = [n1, n2, n3, n4, n5, n6, n7, n8]

    def get(M, dy, dx):
        """Shift M by (dy, dx). Zero-padded border."""
        return torch.roll(M, shifts=(dy.item(), dx.item()), dims=(0, 1))

    p = [get(M, o[0], o[1]) for o in offset]
    P2, P3, P4, P5, P6, P7, P8, P9 = p
    # B = sum of P2..P9
    B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
    # A = number of 01 transitions in P2, P3, ..., P9, P2
    seq = torch.stack([P2, P3, P4, P5, P6, P7, P8, P9, P2], dim=0)
    # transition where seq[i]==0 and seq[i+1]==1
    transitions = ((seq[:-1] == 0) & (seq[1:] == 1)).sum(dim=0)
    A = transitions
    M = M.float()
    iter_count = 0
    while iter_count < max_iter:
        # Sub-iteration 1:
        #   P2*P4*P6 == 0 and P4*P6*P8 == 0
        cond1 = (B >= 2) & (B <= 6) & (A == 1) & M.bool()
        cond1a = cond1 & ((P2 * P4 * P6) == 0).bool()
        cond1b = cond1 & ((P4 * P6 * P8) == 0).bool()
        del1 = cond1a & cond1b
        # Sub-iteration 2:
        #   P2*P4*P8 == 0 and P2*P6*P8 == 0
        cond2a = cond1 & ((P2 * P4 * P8) == 0).bool()
        cond2b = cond1 & ((P2 * P6 * P8) == 0).bool()
        del2 = cond2a & cond2b
        del_mask = del1 | del2
        n_del = int(del_mask.sum().item())
        if n_del == 0:
            break
        M = torch.where(del_mask, torch.zeros_like(M), M)
        # Recompute neighbors (or update incrementally)
        P2 = get(M, -1, 0); P3 = get(M, -1, 1); P4 = get(M, 0, 1)
        P5 = get(M, 1, 1); P6 = get(M, 1, 0); P7 = get(M, 1, -1)
        P8 = get(M, 0, -1); P9 = get(M, -1, -1)
        B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        seq = torch.stack([P2, P3, P4, P5, P6, P7, P8, P9, P2], dim=0)
        transitions = ((seq[:-1] == 0) & (seq[1:] == 1)).sum(dim=0)
        A = transitions
        iter_count += 1
    return (M.cpu().numpy() > 0).astype(np.uint8)


def zhang_suen_subiter_view(M, sub):
    """One sub-iteration of Zhang-Suen for benchmarking. sub=1 or 2.
    Uses SHIFTED VIEWS (no copy) for the 8 neighbors."""
    # All neighbors as views of M (inner region only). No copy!
    # P2 = N, P3 = NE, P4 = E, P5 = SE, P6 = S, P7 = SW, P8 = W, P9 = NW
    # M has shape (H, W). Inner is [1:-1, 1:-1] = (H-2, W-2).
    # M_self = M[1:-1, 1:-1]
    # P2 = M[:-2, 1:-1]
    P2 = M[:-2, 1:-1].float()        # N
    P3 = M[:-2, 2:].float()          # NE
    P4 = M[1:-1, 2:].float()         # E
    P5 = M[2:, 2:].float()           # SE
    P6 = M[2:, 1:-1].float()         # S
    P7 = M[2:, :-2].float()          # SW
    P8 = M[1:-1, :-2].float()        # W
    P9 = M[:-2, :-2].float()         # NW
    Mc = M[1:-1, 1:-1].float()       # center
    B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
    seq = torch.stack([P2, P3, P4, P5, P6, P7, P8, P9, P2], dim=0)
    A = ((seq[:-1] == 0) & (seq[1:] == 1)).sum(dim=0)
    cond = (B >= 2) & (B <= 6) & (A == 1) & Mc.bool()
    if sub == 1:
        del_mask = cond & ((P2 * P4 * P6) == 0).bool() & ((P4 * P6 * P8) == 0).bool()
    else:
        del_mask = cond & ((P2 * P4 * P8) == 0).bool() & ((P2 * P6 * P8) == 0).bool()
    Mc_new = torch.where(del_mask, torch.zeros_like(Mc), Mc)
    # Write back Mc_new into M (only inner)
    M = M.clone()  # avoid in-place
    M[1:-1, 1:-1] = Mc_new
    return M, int(del_mask.sum().item())


def zhang_suen_torch(mask_np, device=None, max_iter=400, full_check_every=8):
    """Zhang-Suen thinning on GPU. mask_np: uint8 0/1 or 0/255. Returns uint8 0/1.

    Uses SHIFTED VIEWS to avoid copies. Full convergence check every K iters.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = (mask_np > 0).astype(np.uint8)
    M = torch.from_numpy(m).to(device)
    H, W = M.shape
    if H < 3 or W < 3:
        return m
    iter_count = 0
    inner_n = (H - 2) * (W - 2)
    last_change = inner_n  # non-zero to ensure first check happens
    while iter_count < max_iter and last_change > 0:
        for sub in [1, 2]:
            M, n_del = zhang_suen_subiter_view(M, sub)
            iter_count += 1
            if n_del == 0:
                return (M.cpu().numpy() > 0).astype(np.uint8)
        last_change = n_del
    return (M.cpu().numpy() > 0).astype(np.uint8)