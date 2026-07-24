"""v35 → v39 final evolution: 100-image eval."""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v35
import v38
import v39

here = Path(__file__).resolve().parent
project = here.parent
MASKS_DIR = project / "yhc" / "mask_refine"


def main():
    files = sorted(MASKS_DIR.glob("*.png"))
    img0 = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    m0 = (img0 > 127).astype(np.uint8)
    _ = v35.fit_v35(m0); _ = v38.fit_v38(m0); _ = v39.fit_v39(m0)
    torch.cuda.synchronize()

    rows = []
    print(f"v35 vs v38 vs v39 on 100 images...")
    for i, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        mask_bin = (img > 127).astype(np.uint8)
        r35 = v35.fit_v35(mask_bin); torch.cuda.synchronize()
        r38 = v38.fit_v38(mask_bin); torch.cuda.synchronize()
        r39 = v39.fit_v39(mask_bin); torch.cuda.synchronize()
        rows.append({"file": f.name,
                     "v35_N": r35["N"], "v35_t": r35["elapsed_s"]["total"] * 1000,
                     "v38_N": r38["N"], "v38_t": r38["elapsed_s"]["total"] * 1000,
                     "v39_N": r39["N"], "v39_t": r39["elapsed_s"]["total"] * 1000,
                     "delta_35_38": r38["N"] - r35["N"],
                     "delta_35_39": r39["N"] - r35["N"]})
        if (i+1) % 25 == 0:
            print(f"  {i+1}/{len(files)}")
    ns35 = np.array([r["v35_N"] for r in rows])
    ns38 = np.array([r["v38_N"] for r in rows])
    ns39 = np.array([r["v39_N"] for r in rows])
    ts35 = np.array([r["v35_t"] for r in rows])
    ts38 = np.array([r["v38_t"] for r in rows])
    ts39 = np.array([r["v39_t"] for r in rows])
    print("\n========== v35 vs v38 vs v39 (100 images) ==========")
    print(f"      N_med  N_mean  N_p95  |  t_med  t_p95")
    print(f"v35:  {np.median(ns35):>4}   {ns35.mean():.1f}   {np.percentile(ns35, 95):>4}    |  {np.median(ts35):>5.0f}  {np.percentile(ts35, 95):>5.0f}")
    print(f"v38:  {np.median(ns38):>4}   {ns38.mean():.1f}   {np.percentile(ns38, 95):>4}    |  {np.median(ts38):>5.0f}  {np.percentile(ts38, 95):>5.0f}")
    print(f"v39:  {np.median(ns39):>4}   {ns39.mean():.1f}   {np.percentile(ns39, 95):>4}    |  {np.median(ts39):>5.0f}  {np.percentile(ts39, 95):>5.0f}")
    delta_39 = ns39 - ns35
    print(f"\ndelta_N(v39 - v35): med={np.median(delta_39):.1f}  "
          f"min={delta_39.min():.0f} max={delta_39.max():.0f}  "
          f"#delta<0: {(delta_39<0).sum()}/100")
    # Real-time fractions
    for v_name, ts in [("v35", ts35), ("v38", ts38), ("v39", ts39)]:
        rt100 = (ts <= 100).mean() * 100
        rt150 = (ts <= 150).mean() * 100
        rt200 = (ts <= 200).mean() * 100
        print(f"{v_name}: <100ms={rt100:.0f}%  <150ms={rt150:.0f}%  <200ms={rt200:.0f}%")
    # Gold standard
    img_004 = cv2.imread(str(project / 'yhc/mask_refine/017_roi_004.png'),
                         cv2.IMREAD_GRAYSCALE)
    m004 = (img_004 > 127).astype(np.uint8)
    print(f"\n[GOLD 017_roi_004] v35 N={v35.fit_v35(m004)['N']}, "
          f"v38 N={v38.fit_v38(m004)['N']}, "
          f"v39 N={v39.fit_v39(m004)['N']}, GT=3")
    import json
    out = project / "algo_v6" / "v39_bench.json"
    out.write_text(json.dumps(rows, indent=2, default=str,
                             ensure_ascii=False))
    print(f"\n[json] {out}")


if __name__ == "__main__":
    main()