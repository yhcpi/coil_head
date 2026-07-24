"""Compare v35 (hard arc_min=0.20) vs v38 (support/r adaptive)."""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v35
import v38

here = Path(__file__).resolve().parent
project = here.parent
MASKS_DIR = project / "yhc" / "mask_refine"


def main():
    files = sorted(MASKS_DIR.glob("*.png"))
    img0 = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    m0 = (img0 > 127).astype(np.uint8)
    _ = v35.fit_v35(m0); _ = v38.fit_v38(m0)
    torch.cuda.synchronize()

    rows = []
    print(f"Processing {len(files)} images...")
    for i, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        mask_bin = (img > 127).astype(np.uint8)
        r35 = v35.fit_v35(mask_bin); torch.cuda.synchronize()
        r38 = v38.fit_v38(mask_bin); torch.cuda.synchronize()
        # Count "good" candidates per v38 (sr > some threshold)
        rows.append({"file": f.name,
                     "v35_N": r35["N"], "v35_t": r35["elapsed_s"]["total"] * 1000,
                     "v38_N": r38["N"], "v38_t": r38["elapsed_s"]["total"] * 1000,
                     "delta_N": r38["N"] - r35["N"]})
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(files)}")
    n35 = np.array([r["v35_N"] for r in rows])
    n38 = np.array([r["v38_N"] for r in rows])
    delta = n38 - n35
    t35 = np.array([r["v35_t"] for r in rows])
    t38 = np.array([r["v38_t"] for r in rows])
    print("\n========== v35 vs v38 (100 images) ==========")
    print(f"v35 N: med={np.median(n35):.0f} mean={n35.mean():.1f} p95={np.percentile(n35, 95):.0f}")
    print(f"v38 N: med={np.median(n38):.0f} mean={n38.mean():.1f} p95={np.percentile(n38, 95):.0f}")
    print(f"delta_N (v38 - v35): med={np.median(delta):.1f}  "
          f"min={delta.min():.0f} max={delta.max():.0f}  "
          f"# images with delta_N<0: {(delta<0).sum()}/100")
    print(f"v35 t: med={np.median(t35):.0f}ms p95={np.percentile(t35, 95):.0f}ms")
    print(f"v38 t: med={np.median(t38):.0f}ms p95={np.percentile(t38, 95):.0f}ms")
    # GT check on 017_roi_004 (gold standard)
    r38_004 = v38.fit_v38((cv2.imread(str(project / 'yhc/mask_refine/017_roi_004.png'),
                                       cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8))
    print(f"\n[GOLD standard 017_roi_004] v35 N=6, v38 N={r38_004['N']}, GT=3")
    if r38_004['N'] == 3:
        print("  ✓✓✓ v38 exactly hits GT=3 on the gold-standard image!")
    import json
    out = project / "algo_v6" / "v38_bench.json"
    out.write_text(json.dumps(rows, indent=2, default=str,
                             ensure_ascii=False))
    print(f"\n[json] {out}")


if __name__ == "__main__":
    main()