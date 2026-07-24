"""Bench v46 (HF + residual check) vs v44 / v39 on 100 images."""
import sys
from pathlib import Path
import numpy as np, cv2, torch
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v39, v44, v46

here = Path(__file__).resolve().parent
project = here.parent
MASKS = project / "yhc" / "mask_refine"


def main():
    files = sorted(MASKS.glob("*.png"))
    m0 = (cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    _ = v39.fit_v39(m0); _ = v44.fit_v44(m0); _ = v46.fit_v46(m0)
    torch.cuda.synchronize()

    rows = []
    print(f"v39/v44/v46 on {len(files)} images...")
    for i, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        m = (img > 127).astype(np.uint8)
        r39 = v39.fit_v39(m); r44 = v44.fit_v44(m); r46 = v46.fit_v46(m)
        torch.cuda.synchronize()
        rows.append({
            "file": f.name,
            "v39_N": r39["N"], "v39_t": r39["elapsed_s"]["total"] * 1000,
            "v44_N": r44["N"], "v44_t": r44["elapsed_s"]["total"] * 1000,
            "v46_N": r46["N"], "v46_t": r46["elapsed_s"]["total"] * 1000,
            "v46_residuals": [c.get("residual_px", 0) for c in r46["circles"]],
        })
        if (i+1) % 25 == 0: print(f"  {i+1}/{len(files)}")

    for tag in ["v39", "v44", "v46"]:
        ns = np.array([r[f"{tag}_N"] for r in rows])
        ts = np.array([r[f"{tag}_t"] for r in rows])
        print(f"\n{tag}: N_med={np.median(ns):.0f} N_mean={ns.mean():.1f}  "
              f"t_med={np.median(ts):.0f}ms  p95={np.percentile(ts,95):.0f}ms")
    # residual distribution
    all_res = [r for row in rows for r in row["v46_residuals"]]
    if all_res:
        all_res = np.array(all_res)
        print(f"\nv46 residual_px: med={np.median(all_res):.3f} "
              f"p95={np.percentile(all_res,95):.3f} max={all_res.max():.3f}")
    # Gold
    g = cv2.imread(str(project / "yhc/mask_refine/017_roi_004.png"),
                   cv2.IMREAD_GRAYSCALE)
    gm = (g > 127).astype(np.uint8)
    print(f"\nGold 017_004: v39={v39.fit_v39(gm)['N']}  "
          f"v44={v44.fit_v44(gm)['N']}  v46={v46.fit_v46(gm)['N']}  GT=3")
    (project / "algo_v6" / "v46_bench.json").write_text(
        json.dumps(rows, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()