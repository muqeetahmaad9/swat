"""Regenerate the SegFormer XAI figures with the dark style, pointing at the
16-band cube the model was trained on (the live cube is now 36-band)."""
import sys

import xai_analysis as xa
import segformer_flood as sf

sf.CFG["cube_path"]  = "../data/processed/feature_cube_16band.npy"
sf.CFG["valid_path"] = "../data/processed/valid_mask_16band.npy"
xa.XCFG["meta_path"] = "../data/processed/metadata_16band.json"

import dl_xai

mode = sys.argv[1] if len(sys.argv) > 1 else "igcam"
names = dl_xai.band_names()
model, ds, tiles, ys, ms, labels = dl_xai.load_all()
if mode == "igcam":
    dl_xai.map_comparison(labels)
    dl_xai.gradcam_figure(model, ds, names)
    dl_xai.integrated_gradients(model, tiles, names)
else:
    dl_xai.permutation(model, tiles, ys, ms, names)
    dl_xai.consistency(names)
print("done:", mode)
