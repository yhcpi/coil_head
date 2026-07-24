"""Find outlier cases where HF wildly differs from cv2."""
import sys
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v32, v35, v39, ellipse_halir_flusser as hf

project = Path(__file__).resolve().parent.parent
MASKS_DIR = project / "yhc" / "mask_refine"

# Use v39's full pipeline so we match the inlier sets it actually uses
for f in sorted(MASKS_DIR.glob("*.png"))[:30]:
    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    mask_bin = (img > 127).astype(np.uint8)
    res39 = v39.fit_v39(mask_bin)
    for c39 in res39["circles"]:
        # Re-run on a synthetic near-circle matching this r
        # Actually just check c39's parameters
        r = c39["r"]
        # Re-extract inliers by sampling points on the ellipse
        cx, cy = c39["cx"], c39["cy"]
        a, b = c39["a"], c39["b"]
        ang = c39["ang"]
        ar = np.radians(ang)
        t = np.linspace(0, 2*np.pi, 200)
        xloc = a*np.cos(t); yloc = b*np.sin(t)
        x = cx + np.cos(ar)*xloc - np.sin(ar)*yloc
        y = cy + np.sin(ar)*xloc + np.cos(ar)*yloc
        pts = np.column_stack([x, y])
        ref_cv = v32._fit_ellipse_safe(pts.reshape(-1, 1, 2))
        ref_hf = hf.halir_flusser_fit(pts)
        if ref_cv and ref_hf:
            diff_a = abs(ref_cv[2] - ref_hf[2])
            if diff_a > 5:
                print(f"OVERSIZE {f.name}: r={r:.1f} cv_a={ref_cv[2]:.1f} hf_a={ref_hf[2]:.1f} "
                      f"diff={diff_a:.1f}")