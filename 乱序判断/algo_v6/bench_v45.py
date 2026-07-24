"""Bench v45 (adaptive RANSAC) vs v39 (fixed 12 rounds) on 100 images."""
import sys
from pathlib import Path
import numpy as np, cv2, torch
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v39, v45

here = Path(__file__).resolve().parent
project = here.parent
MASKS = project / "yhc" / "mask_refine"


def main():
    files = sorted(MASKS.glob("*.png"))
    m0 = (cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    _ = v39.fit_v39(m0); _ = v45.fit_v45(m0); torch.cuda.synchronize()

    rows = []
    print(f"v39 vs v45 (adaptive RANSAC) on {len(files)} images...")
    for i, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        m = (img > 127).astype(np.uint8)
        r39 = v39.fit_v39(m); torch.cuda.synchronize()
        r45 = v45.fit_v45(m); torch.cuda.synchronize()
        rows.append({
            "file": f.name,
            "v39_N": r39["N"], "v39_t": r39["elapsed_s"]["total"] * 1000,
            "v45_N": r45["N"], "v45_t": r45["elapsed_s"]["total"] * 1000,
            "v45_rounds_used": r45.get("rounds_used", 0),
        })
        if (i+1) % 25 == 0: print(f"  {i+1}/{len(files)}")

    n39 = np.array([r["v39_N"] for r in rows])
    n45 = np.array([r["v45_N"] for r in rows])
    t39 = np.array([r["v39_t"] for r in rows])
    t45 = np.array([r["v45_t"] for r in rows])
    print(f"\n========== v39 vs v45 (adaptive) ==========")
    print(f"v39: N_med={np.median(n39):.0f} N_mean={n39.mean():.1f}  t_med={np.median(t39):.0f}ms")
    print(f"v45: N_med={np.median(n45):.0f} N_mean={n45.mean():.1f}  t_med={np.median(t45):.0f}ms")
    print(f"speedup: {(np.median(t39)/np.median(t45) - 1)*100:+.1f}%  "
          f"|ΔN|<=1: {(np.abs(n45 - n39) <= 1).sum()}/100  "
          f"v45>v39: {(n45 > n39).sum()}  v39>v45: {(n45 < n39).sum()}")
    rounds = [r["v45_rounds_used"] for r in rows]
    print(f"v45 rounds used: med={np.median(rounds):.1f}, p10={np.percentile(rounds,10):.0f}, "
          f"max={max(rounds)} (cap=12)")
    # Gold
    g = cv2.imread(str(project / "yhc/mask_refine/017_roi_004.png"),
                   cv2.IMREAD_GRAYSCALE)
    gm = (g > 127).astype(np.uint8)
    print(f"Gold 017_004: v39 N={v39.fit_v39(gm)['N']}  v45 N={v45.fit_v45(gm)['N']}  GT=3")
    (project / "algo_v6" / "v45_bench.json").write_text(
        json.dumps(rows, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()