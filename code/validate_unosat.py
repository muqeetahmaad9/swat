"""
Independent flood-label validation against UNOSAT (reviewer item R1.2).

Reference: UNOSAT VIIRS-derived flood extents for Pakistan, Aug 2022
(FL20220808PAK product, CERN mirror). The 25-31 Aug layer brackets the KP
flood peak (25-27 Aug); the 01-31 Aug cumulative layer is used as a
sensitivity check. VIIRS is 375 m native vs our 30 m Sentinel-1 labels, so
pixel-perfect agreement is impossible by construction; metrics quantify
consistency, not identity.

Protocol: clip UNOSAT polygons to the AOI -> rasterize onto the 30 m analysis
grid -> compare against flood_labels_2022 within the floodable domain,
excluding our 'unknown' pixels. Reports IoU, Cohen's kappa, and the two
directional agreements. Also reports both at 375 m block aggregation
(any-flood within 13x13 blocks), the fairer scale for a VIIRS reference.
"""

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import box

BASE = "../data/unosat/FL20220808PAK_SHP/"
LAYERS = ["VIIRS_20220825_20220831_FloodExtent_PAK",
          "VIIRS_20220801_20220831_FloodExtent_PAK"]
AOI = box(71.0, 34.0, 72.9, 35.95)


def block_any(a, k=13):
    """Aggregate a boolean grid to k x k blocks with logical OR."""
    H, W = a.shape
    Hc, Wc = H // k * k, W // k * k
    return a[:Hc, :Wc].reshape(Hc // k, k, Wc // k, k).any(axis=(1, 3))


def kappa(y1, y2):
    po = (y1 == y2).mean()
    p1, p2 = y1.mean(), y2.mean()
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (po - pe) / (1 - pe + 1e-12)


def main():
    with rasterio.open("../data/labels/flood_labels_2022.tif") as src:
        lab = src.read(1)
        transform, shape = src.transform, (src.height, src.width)

    ours = lab == 1                     # our flood
    known = lab != 255                  # inside floodable domain & not unknown

    for layer in LAYERS:
        g = gpd.read_file(BASE + layer + ".shp")
        clipped = gpd.clip(g.to_crs("EPSG:4326"), AOI).to_crs("EPSG:32643")
        if clipped.empty or clipped.geometry.is_empty.all():
            print(f"{layer}: no coverage in AOI")
            continue
        area = clipped.area.sum() / 1e6
        ref = rasterize(((geom, 1) for geom in clipped.geometry),
                        out_shape=shape, transform=transform,
                        fill=0, dtype="uint8").astype(bool)
        print(f"\n=== {layer} ===")
        print(f"UNOSAT flood in AOI: {area:.1f} km2 "
              f"({int(ref.sum()):,} px on 30 m grid)")
        print(f"our flood (2022):    {int(ours.sum())*0.0009:.1f} km2")

        m = known                                        # comparable pixels
        a, b = ours[m], ref[m]
        inter, union = (a & b).sum(), (a | b).sum()
        print(f"[30 m,  n={m.sum():,}] IoU {inter/max(union,1):.3f} | "
              f"kappa {kappa(a, b):.3f} | "
              f"P(ours|UNOSAT) {(a & b).sum()/max(b.sum(),1):.3f} | "
              f"P(UNOSAT|ours) {(a & b).sum()/max(a.sum(),1):.3f}")

        # fairer scale: 375 m blocks (13 x 30 m), flood = any flood in block
        ob, rb = block_any(ours & known), block_any(ref & known)
        kb = block_any(known)                            # blocks with any data
        a, b = ob[kb], rb[kb]
        inter, union = (a & b).sum(), (a | b).sum()
        print(f"[375 m, n={kb.sum():,}] IoU {inter/max(union,1):.3f} | "
              f"kappa {kappa(a, b):.3f} | "
              f"P(ours|UNOSAT) {(a & b).sum()/max(b.sum(),1):.3f} | "
              f"P(UNOSAT|ours) {(a & b).sum()/max(a.sum(),1):.3f}")


if __name__ == "__main__":
    main()
