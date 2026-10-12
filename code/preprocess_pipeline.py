"""
Preprocessing pipeline: GEE GeoTIFF exports  ->  ML/DL training dataset.

v2 — disk-backed assembly. The enriched feature set (~34 encoded bands over a
7390x6005 grid) no longer fits comfortably in RAM, so the cube is written
band-by-band into a numpy memmap (`feature_cube.npy`) instead of being stacked
in memory. Everything downstream (samplers, trainers) already reads the same
file and keeps working.

Steps: discover/mosaic -> reference grid -> per-layer reproject -> clip ->
encode (z-score / sin-cos / one-hot) straight into the memmap -> validity
mask -> median fill -> sample points -> spatial block split -> pixel vectors
+ CNN patches -> export + metadata.

Usage: python preprocess_pipeline.py
"""

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.warp import reproject

# ============================================================================
CONFIG = {
    "raw_dir": "../data/raw",
    "label_path": "../data/labels/flood_labels.tif",
    "aoi_path": None,
    "out_dir": "../data/processed",
    "reference_layer": "swat_dem",
    "categorical": ["swat_lulc"],          # one-hot encoded, nearest resampling
    "circular": ["swat_aspect"],           # sin/cos encoded
    "patch_size": 32,
    "n_neg_ratio": 1.0,
    "max_pos": 10000,                      # memory guard on flood samples
    "block_size_px": 200,                  # 6 km spatial-CV blocks
    "split_fracs": (0.7, 0.15, 0.15),
    "seed": 42,
}

# ============================================================================
# Discovery & IO
# ============================================================================

def discover_layers(raw_dir):
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.tif"))):
        base = re.sub(r"(-\d{10})+$", "", os.path.basename(path)[:-4])
        groups[base].append(path)
    if not groups:
        raise FileNotFoundError(f"No .tif files found in {raw_dir}")
    return dict(groups)


def read_mosaic(paths):
    srcs = [rasterio.open(p) for p in paths]
    if len(srcs) == 1:
        arr, profile = srcs[0].read(), srcs[0].profile.copy()
    else:
        arr, transform = rio_merge(srcs)
        profile = srcs[0].profile.copy()
        profile.update(height=arr.shape[1], width=arr.shape[2], transform=transform)
    nodata = profile.get("nodata")
    arr = arr.astype("float32")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    for s in srcs:
        s.close()
    return arr, profile


def get_reference(layers, ref_name):
    arr, profile = read_mosaic(layers[ref_name])
    return {"crs": profile["crs"], "transform": profile["transform"],
            "height": arr.shape[1], "width": arr.shape[2]}


def resample_to_ref(arr, profile, ref, method):
    out = np.full((arr.shape[0], ref["height"], ref["width"]), np.nan, "float32")
    for b in range(arr.shape[0]):
        reproject(source=arr[b], destination=out[b],
                  src_transform=profile["transform"], src_crs=profile["crs"],
                  dst_transform=ref["transform"], dst_crs=ref["crs"],
                  resampling=method, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def basin_mask(ref, aoi_path):
    if aoi_path is None:
        return np.ones((ref["height"], ref["width"]), bool)
    import geopandas as gpd
    from rasterio.features import rasterize
    gdf = gpd.read_file(aoi_path).to_crs(ref["crs"])
    return rasterize([(g, 1) for g in gdf.geometry],
                     out_shape=(ref["height"], ref["width"]),
                     transform=ref["transform"], fill=0,
                     dtype="uint8").astype(bool)

# ============================================================================
# Pass 1 — plan the encoded band layout (names only; cheap)
# ============================================================================

def plan_bands(layers, cfg, ref):
    """Return ordered list of (layer, kind, subinfo, band_name)."""
    plan = []
    for name, paths in layers.items():
        if name in cfg["circular"]:
            plan.append((name, "circ_sin", None, f"{name}_sin"))
            plan.append((name, "circ_cos", None, f"{name}_cos"))
        elif name in cfg["categorical"]:
            arr, profile = read_mosaic(paths)                 # read once for classes
            arr = resample_to_ref(arr, profile, ref, Resampling.nearest)
            classes = np.unique(arr[0][~np.isnan(arr[0])]).astype(int)
            for c in classes:
                plan.append((name, "onehot", int(c), f"{name}_cls{c}"))
            del arr
        else:
            with rasterio.open(paths[0]) as src:
                nb = src.count
            for b in range(nb):
                plan.append((name, "cont", b,
                             f"{name}_b{b}" if nb > 1 else name))
    return plan

# ============================================================================
# Pass 2 — fill the memmap cube band by band
# ============================================================================

def build_cube(layers, plan, cfg, ref, inside):
    H, W = ref["height"], ref["width"]
    C = len(plan)
    cube_path = os.path.join(cfg["out_dir"], "feature_cube.npy")
    cube = np.lib.format.open_memmap(cube_path, mode="w+",
                                     dtype="float32", shape=(H, W, C))
    valid = inside.copy()
    stats = {}
    ci = 0
    for name, paths in layers.items():                        # layer by layer
        method = (Resampling.nearest
                  if name in cfg["categorical"] + cfg["circular"]
                  else Resampling.bilinear)
        arr, profile = read_mosaic(paths)
        arr = resample_to_ref(arr, profile, ref, method)
        arr[:, ~inside] = np.nan
        entries = [p for p in plan if p[0] == name]
        for (_, kind, sub, bname) in entries:
            if kind == "circ_sin":
                band = np.sin(np.radians(arr[0]))
            elif kind == "circ_cos":
                band = np.cos(np.radians(arr[0]))
            elif kind == "onehot":
                band = (arr[0] == sub).astype("float32")
                band[np.isnan(arr[0])] = np.nan
            else:                                             # continuous, z-score
                band = arr[sub]
                mean = float(np.nanmean(band)); std = float(np.nanstd(band)) or 1.0
                stats[bname] = {"mean": mean, "std": std}
                band = (band - mean) / std
            valid &= np.isfinite(band)                        # accumulate validity
            med = np.nanmedian(band)                          # median fill this band
            band = np.nan_to_num(band, nan=med).astype("float32")
            cube[:, :, ci] = band
            ci += 1
        del arr
        print(f"[3-6] {name}: {len(entries)} band(s) written "
              f"({ci}/{C})", flush=True)
    cube.flush()
    return cube, valid, stats

# ============================================================================
# Sampling, splitting, patches (unchanged logic)
# ============================================================================

def sample_points(labels, valid, ratio, rng, max_pos=None):
    pos = np.argwhere((labels == 1) & valid)
    if max_pos is not None and len(pos) > max_pos:
        pos = pos[rng.choice(len(pos), max_pos, replace=False)]
    neg_pool = np.argwhere((labels == 0) & valid)
    n_neg = min(int(len(pos) * ratio), len(neg_pool))
    neg = neg_pool[rng.choice(len(neg_pool), n_neg, replace=False)]
    pts = np.vstack([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype("int8")
    return pts, y


def block_split(points, shape, block_px, fracs, rng):
    n_bcols = int(np.ceil(shape[1] / block_px))
    block_id = (points[:, 0] // block_px) * n_bcols + (points[:, 1] // block_px)
    uniq = rng.permutation(np.unique(block_id))
    n_tr = int(len(uniq) * fracs[0]); n_va = int(len(uniq) * fracs[1])
    sets = {"train": set(uniq[:n_tr]), "val": set(uniq[n_tr:n_tr + n_va]),
            "test": set(uniq[n_tr + n_va:])}
    return {k: np.isin(block_id, list(v)) for k, v in sets.items()}


def extract_patches(cube, points, y, patch):
    half = patch // 2
    H, W = cube.shape[:2]
    keep = ((points[:, 0] >= half) & (points[:, 0] < H - half) &
            (points[:, 1] >= half) & (points[:, 1] < W - half))
    pts, y = points[keep], y[keep]
    X = np.stack([np.asarray(cube[r - half:r + half, c - half:c + half, :])
                  for r, c in pts]).astype("float32")
    return X, y, pts

# ============================================================================
# MAIN
# ============================================================================

def main(cfg=CONFIG):
    rng = np.random.default_rng(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    layers = discover_layers(cfg["raw_dir"])
    print(f"[1] Found {len(layers)} layers: {list(layers)}")
    ref = get_reference(layers, cfg["reference_layer"])
    print(f"[2] Reference grid: {ref['width']}x{ref['height']} px, {ref['crs']}")

    inside = basin_mask(ref, cfg["aoi_path"])
    plan = plan_bands(layers, cfg, ref)
    band_names = [p[3] for p in plan]
    print(f"[plan] {len(plan)} encoded bands")

    cube, valid, stats = build_cube(layers, plan, cfg, ref, inside)
    np.save(os.path.join(cfg["out_dir"], "valid_mask.npy"), valid)
    print(f"[7] Feature cube: {cube.shape}, valid pixels: {valid.sum():,}")

    if os.path.exists(cfg["label_path"]):
        with rasterio.open(cfg["label_path"]) as src:
            lab = np.full((ref["height"], ref["width"]), np.nan, "float32")
            reproject(source=src.read(1).astype("float32"), destination=lab,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref["transform"], dst_crs=ref["crs"],
                      resampling=Resampling.nearest,
                      src_nodata=src.nodata, dst_nodata=np.nan)

        points, y = sample_points(lab, valid, cfg["n_neg_ratio"], rng,
                                  cfg.get("max_pos"))
        print(f"[8] Samples: {int((y==1).sum()):,} flood / {int((y==0).sum()):,} non-flood")

        splits = block_split(points, (ref["height"], ref["width"]),
                             cfg["block_size_px"], cfg["split_fracs"], rng)
        for split, m in splits.items():
            pts_s, y_s = points[m], y[m]
            Xv = np.asarray(cube[pts_s[:, 0], pts_s[:, 1], :])
            np.save(os.path.join(cfg["out_dir"], f"X_{split}_pixels.npy"), Xv)
            np.save(os.path.join(cfg["out_dir"], f"y_{split}_pixels.npy"), y_s)
            Xp, yp, pts_kept = extract_patches(cube, pts_s, y_s, cfg["patch_size"])
            np.save(os.path.join(cfg["out_dir"], f"X_{split}_patches.npy"), Xp)
            np.save(os.path.join(cfg["out_dir"], f"y_{split}.npy"), yp)
            np.save(os.path.join(cfg["out_dir"], f"points_{split}.npy"), pts_kept)
            print(f"[9] {split}: pixels {Xv.shape}, patches {Xp.shape}", flush=True)
    else:
        print("[8] No label raster -> feature cube only.")

    meta = {"band_names": band_names, "normalization_stats": stats,
            "crs": str(ref["crs"]), "transform": list(ref["transform"])[:6],
            "shape": [ref["height"], ref["width"]],
            "patch_size": cfg["patch_size"], "block_size_px": cfg["block_size_px"],
            "seed": cfg["seed"]}
    with open(os.path.join(cfg["out_dir"], "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[9] Done. Outputs in {os.path.abspath(cfg['out_dir'])}")


if __name__ == "__main__":
    main()
