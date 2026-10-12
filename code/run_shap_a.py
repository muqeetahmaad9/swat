"""Run only Part A (TreeSHAP on XGBoost) of xai_analysis — the DL parts need a
16-channel checkpoint and cannot run against the 36-band cube."""
from xai_analysis import part_a_treeshap, band_names

part_a_treeshap(band_names())
