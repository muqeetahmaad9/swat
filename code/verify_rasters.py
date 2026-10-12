"""Sanity-check the GEE exports in data/raw: grid consistency, CRS, values."""
import glob, os
import numpy as np
import rasterio

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

rows, ref = [], None
for path in sorted(glob.glob(os.path.join(RAW, "*.tif"))):
    with rasterio.open(path) as src:
        # subsample for speed: read every 8th pixel of band 1
        a = src.read(1, out_shape=(src.height // 8, src.width // 8)).astype("float64")
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        finite = np.isfinite(a)
        rows.append({
            "name": os.path.basename(path),
            "crs": str(src.crs),
            "size": f"{src.width}x{src.height}",
            "res": f"{src.res[0]:.1f}m",
            "bands": src.count,
            "nodata": src.nodata,
            "min": np.nanmin(a) if finite.any() else float("nan"),
            "max": np.nanmax(a) if finite.any() else float("nan"),
            "mean": np.nanmean(a) if finite.any() else float("nan"),
            "pct_nodata": 100 * (1 - finite.mean()),
        })
        if ref is None:
            ref = (str(src.crs), src.width, src.height, src.transform)
        same = (str(src.crs), src.width, src.height, src.transform) == ref
        rows[-1]["grid_match"] = "OK" if same else "MISMATCH"

hdr = f"{'file':<26}{'crs':<13}{'size':<13}{'res':<8}{'bands':<6}{'min':>10}{'max':>10}{'mean':>10}{'%nodata':>9}  grid"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['name']:<26}{r['crs']:<13}{r['size']:<13}{r['res']:<8}{r['bands']:<6}"
          f"{r['min']:>10.2f}{r['max']:>10.2f}{r['mean']:>10.2f}{r['pct_nodata']:>8.1f}%  {r['grid_match']}")
