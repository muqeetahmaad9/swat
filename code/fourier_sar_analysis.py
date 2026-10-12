"""
Frequency-domain (2-D Fourier) characterization of the Sentinel-1 VH image.

Why this exists
---------------
An independent, physics-based cross-check of a result the paper already
reaches by SHAP and permutation importance: that Sentinel-1 VH backscatter
separates flooded from dry floodable terrain. Instead of bolting on an
unrelated technique, we decompose the SAR image into spatial frequencies and
ask WHERE in the spectrum the flood/dry information lives.

What the data say (honest result)
---------------------------------
The 2-D Fourier transform of a tile splits its content into a zero-frequency
(DC) term -- which is exactly the tile mean backscatter -- and higher
wavenumbers, which encode spatial texture. Across ~150 flood-dominated tiles:

  * the DC / mean-level term separates the classes strongly
    (flood ~ -18.6 dB vs dry ~ -17.4 dB, Mann-Whitney p ~ 1e-20);
  * the higher-frequency texture does NOT separate them
    (high-frequency energy fraction, p ~ 1).

So the discriminating information is RADIOMETRIC (the specular darkening of
smooth water), not TEXTURAL. This confirms, from an entirely different
direction, the SHAP dependence finding that VH acts through its absolute
level near the -17 dB inventory threshold -- and it explains why single-look
SAR texture features would add little for this problem.

Pipeline: rasterio (geospatial library) to read the co-registered VH and
label rasters -> tile -> numpy.fft.fft2 -> radial spectrum + descriptors ->
Mann-Whitney tests -> fig_fourier_sar.

Run:  python fourier_sar_analysis.py
"""

import os
import numpy as np
import rasterio
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# --------------------------------------------------------------------------- #
# paths / parameters                                                          #
# --------------------------------------------------------------------------- #
VH_PATH  = "../data/raw/swat_s1_vh.tif"
LAB_PATH = "../data/labels/flood_labels.tif"
FIG_DIR  = "../figures"

TILE       = 32            # 32 px = 0.96 km at 30 m; yields ~150 flood tiles
FLOOD_FRAC = 0.30          # tile is "flooded" if >= 30 % of pixels are flood
DRY_FRAC   = 0.50          # "dry-floodable" if 0 % flood and >= 50 % label-0
VH_LO, VH_HI = -35.0, 5.0  # plausible dB range; anything else is nodata/off-swath
MAX_DRY    = 500           # cap dry tiles for balanced, fast statistics
SEED       = 0

# --------------------------------------------------------------------------- #
# figure style -- self-contained copy of the paper's palette                  #
# --------------------------------------------------------------------------- #
INK, MUTED, GRID = "#111111", "#333333", "#dcdcdc"
C_FLOOD, C_DRY   = "#0072B2", "#E69F00"     # Okabe-Ito blue / orange (CVD-safe)
SEQ_CMAP         = "viridis"

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 400,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5, "axes.titlesize": 10.0, "axes.labelsize": 9.5,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.labelcolor": INK,
    "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

# --------------------------------------------------------------------------- #
# spectral helpers (2-D FFT geometry precomputed once)                        #
# --------------------------------------------------------------------------- #
_WIN2D = np.outer(np.hanning(TILE), np.hanning(TILE))   # separable Hann window
_cy = _cx = TILE // 2
_yy, _xx = np.mgrid[0:TILE, 0:TILE]
_R = np.round(np.hypot(_yy - _cy, _xx - _cx)).astype(int)
_KMAX = TILE // 2
_MID = _KMAX // 2                                        # low-|-high k boundary


def power_spectrum(tile):
    """2-D power spectrum after mean removal + Hann windowing (the window
    kills the wrap-around edge that would inject spurious high-k energy)."""
    t = (tile - tile.mean()) * _WIN2D
    F = np.fft.fftshift(np.fft.fft2(t))
    return F.real ** 2 + F.imag ** 2


def radial_profile(P):
    """Azimuthally averaged power per integer radial wavenumber k."""
    tot = np.bincount(_R.ravel(), P.ravel())
    cnt = np.bincount(_R.ravel())
    return tot[:_KMAX + 1] / np.maximum(cnt[:_KMAX + 1], 1)


def hf_energy_fraction(P):
    """Share of AC (non-DC) spectral energy at high wavenumbers (k >= KMAX/2).
    Higher = more fine-scale texture."""
    energy = np.bincount(_R.ravel(), P.ravel())[:_KMAX + 1]
    ac = energy[1:]
    return ac[_MID:].sum() / max(ac.sum(), 1e-12)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(SEED)

    with rasterio.open(VH_PATH) as ds:
        vh = ds.read(1).astype(np.float32)
    with rasterio.open(LAB_PATH) as ds:
        lab = ds.read(1)
    assert vh.shape == lab.shape, "VH and label rasters are not co-registered"
    print(f"VH raster {vh.shape} | labels {lab.shape} | rasterio, EPSG:32643")

    H, W = vh.shape
    flood, dry = [], []
    for r in range(0, H - TILE + 1, TILE):
        for c in range(0, W - TILE + 1, TILE):
            v = vh[r:r + TILE, c:c + TILE]
            if not np.isfinite(v).all() or v.min() < VH_LO or v.max() > VH_HI:
                continue
            l = lab[r:r + TILE, c:c + TILE]
            ff = np.mean(l == 1)
            if ff >= FLOOD_FRAC:
                flood.append((r, c))
            elif ff == 0.0 and np.mean(l == 0) >= DRY_FRAC:
                dry.append((r, c))
    dry = [dry[i] for i in rng.permutation(len(dry))[:MAX_DRY]]
    print(f"tiles: flooded {len(flood)} | dry-floodable {len(dry)}")

    def measure(tiles):
        prof, mean_db, hfef = [], [], []
        for (r, c) in tiles:
            t = vh[r:r + TILE, c:c + TILE]
            P = power_spectrum(t)
            prof.append(radial_profile(P))
            mean_db.append(float(t.mean()))
            hfef.append(hf_energy_fraction(P))
        return np.array(prof), np.array(mean_db), np.array(hfef)

    fp, fm, fh = measure(flood)
    dp, dm, dh = measure(dry)

    # radial curves, normalized so we compare spectral SHAPE not level
    def curve_db(prof):
        m = prof.mean(0); m = m / m[1:].sum()
        return 10.0 * np.log10(np.maximum(m, 1e-12))
    fc, dc = curve_db(fp), curve_db(dp)
    k = np.arange(_KMAX + 1)

    _, p_mean = mannwhitneyu(fm, dm)          # DC / mean-level separation
    _, p_hfef = mannwhitneyu(fh, dh)          # texture separation
    print("\n--- Fourier descriptors: flood vs dry-floodable --------------")
    print(f"  DC / mean VH (dB)  flood {fm.mean():7.2f} | dry {dm.mean():7.2f}"
          f"  p={p_mean:.1e}")
    print(f"  HF energy fraction flood {fh.mean():7.3f} | dry {dh.mean():7.3f}"
          f"  p={p_hfef:.1e}")
    print("--------------------------------------------------------------")

    # representative example tiles: nearest to each class mean backscatter
    f_ex = flood[int(np.argmin(np.abs(fm - fm.mean())))]
    d_ex = dry[int(np.argmin(np.abs(dm - dm.mean())))]
    f_img = vh[f_ex[0]:f_ex[0] + TILE, f_ex[1]:f_ex[1] + TILE]
    d_img = vh[d_ex[0]:d_ex[0] + TILE, d_ex[1]:d_ex[1] + TILE]
    f_ps, d_ps = power_spectrum(f_img), power_spectrum(d_img)

    # ---------------------------------------------------------------- figure --
    fig, ax = plt.subplots(2, 3, figsize=(9.4, 6.1))
    for a in ax.ravel():
        a.grid(False)

    def tile_panel(a, img, title):
        a.imshow(img, cmap="Greys_r", vmin=VH_LO, vmax=0)
        a.set_title(title, fontsize=9.5)
        a.set_xticks([]); a.set_yticks([])

    def spec_panel(a, P, title):
        a.imshow(np.log10(P + 1e-6), cmap=SEQ_CMAP)
        a.set_title(title, fontsize=9.5)
        a.set_xticks([]); a.set_yticks([])

    tile_panel(ax[0, 0], f_img, "Flooded tile — VH (dB)")
    spec_panel(ax[0, 1], f_ps, "Flooded — 2-D power spectrum")
    tile_panel(ax[1, 0], d_img, "Dry-floodable tile — VH (dB)")
    spec_panel(ax[1, 1], d_ps, "Dry-floodable — 2-D power spectrum")

    # top-right: radial AC spectra -- these nearly overlap (texture is similar)
    a = ax[0, 2]; a.grid(True)
    a.plot(k[1:], fc[1:], color=C_FLOOD, lw=1.8, label="Flooded")
    a.plot(k[1:], dc[1:], color=C_DRY, lw=1.8, label="Dry-floodable")
    a.set_xlabel("radial wavenumber $k$ (cycles / tile)")
    a.set_ylabel("normalized AC power (dB)")
    a.set_title(f"Texture spectra overlap\n(high-$k$ energy $p={p_hfef:.2f}$)",
                fontsize=9.5)
    a.legend(loc="upper right", fontsize=8.5)

    # bottom-right: DC / mean-level -- this is where the classes separate
    a = ax[1, 2]; a.grid(True, axis="y")
    parts = a.violinplot([fm, dm], positions=[0, 1], showmeans=True,
                         showextrema=False, widths=0.8)
    for b, col in zip(parts["bodies"], (C_FLOOD, C_DRY)):
        b.set_facecolor(col); b.set_edgecolor(col); b.set_alpha(0.45)
    parts["cmeans"].set_color(INK); parts["cmeans"].set_linewidth(1.3)
    a.axhline(-17.0, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    a.text(1.46, -17.0, "  −17 dB\n  inventory\n  threshold", va="center",
           ha="left", fontsize=7.2, color=MUTED)
    a.set_xticks([0, 1]); a.set_xticklabels(["Flooded", "Dry-\nfloodable"])
    a.set_ylabel("DC term = tile mean VH (dB)")
    a.set_title(f"Mean level separates\n($p<${max(p_mean,1e-300):.0e})",
                fontsize=9.5)
    a.set_xlim(-0.6, 2.6)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "fig_fourier_sar.png"))
    fig.savefig(os.path.join(FIG_DIR, "fig_fourier_sar.pdf"))
    plt.close(fig)
    print("  figure saved: fig_fourier_sar.png / .pdf")

    print("\nMANUSCRIPT NUMBERS")
    print(f"  n_flood={len(flood)} n_dry={len(dry)} tile={TILE}px "
          f"({TILE*30/1000:.2f} km)")
    print(f"  DC mean VH: flood {fm.mean():.1f} dB, dry {dm.mean():.1f} dB, "
          f"p={p_mean:.0e}")
    print(f"  HFEF: flood {fh.mean():.3f}, dry {dh.mean():.3f}, p={p_hfef:.2f}")


if __name__ == "__main__":
    main()
