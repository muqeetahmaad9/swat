"""Export the published susceptibility products and original-split sample
points into the public repository (repository/data/).

- susceptibility_xgb_prob_uint8.tif : P(flood) x 250 as uint8 (0-250), 255 = not
  assessed (outside the floodable domain), deflate-compressed
- susceptibility_xgb_classes.tif    : quintile classes 1-5 (very low .. very high)
  within the floodable domain, 0 = not assessed (as Fig. 7)
- samples/: original-split point coordinates (row, col on the 30 m grid) and labels
"""
import json
import os
import shutil

import numpy as np
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SWAT_DATA", r"C:\Users\4975osama\Desktop\paper\data")
OUT = os.path.join(HERE, "..", "repository", "data")
os.makedirs(os.path.join(OUT, "susceptibility"), exist_ok=True)
os.makedirs(os.path.join(OUT, "samples"), exist_ok=True)

with rasterio.open(os.path.join(DATA, "processed", "susceptibility_xgb.tif")) as src:
    prob = src.read(1).astype("float32")
    prof = src.profile.copy()
with rasterio.open(os.path.join(DATA, "labels", "flood_labels.tif")) as src:
    lab = src.read(1)
    nod = src.nodata
floodable = (lab != 255) & ((lab != nod) if nod is not None else True)
prob = np.where(floodable & np.isfinite(prob), prob, np.nan)

vals = prob[np.isfinite(prob)]
breaks = np.nanquantile(vals, [0.2, 0.4, 0.6, 0.8])
classes = np.where(np.isfinite(prob), np.digitize(prob, breaks) + 1, 0).astype("uint8")
p8 = np.where(np.isfinite(prob), np.clip(np.round(prob * 250), 0, 250), 255).astype("uint8")

prof.update(dtype="uint8", count=1, compress="deflate", predictor=2, tiled=True,
            blockxsize=512, blockysize=512)
for name, arr, nodata, desc in [
        ("susceptibility_xgb_prob_uint8.tif", p8, 255, "P(flood)*250; 255 = not assessed"),
        ("susceptibility_xgb_classes.tif", classes, 0, "1 very low ... 5 very high; 0 = not assessed")]:
    prof.update(nodata=nodata)
    path = os.path.join(OUT, "susceptibility", name)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, desc)
    print(f"{name}: {os.path.getsize(path) / 1e6:.1f} MB")

km2 = {int(c): float((classes == c).sum() * 900 / 1e6) for c in range(1, 6)}
json.dump({"class_breaks_probability": [float(b) for b in breaks],
           "class_area_km2": km2,
           "floodable_domain_km2": float(np.isfinite(prob).sum() * 900 / 1e6)},
          open(os.path.join(OUT, "susceptibility", "classes_info.json"), "w"), indent=2)
print("class areas km2:", km2)

for f in ["points_train.npy", "points_val.npy", "points_test.npy",
          "y_train.npy", "y_val.npy", "y_test.npy", "metadata.json"]:
    shutil.copy(os.path.join(DATA, "processed", f), os.path.join(OUT, "samples", f))
print("samples copied")
