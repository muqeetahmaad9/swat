# Flood susceptibility in the Swat Basin, Pakistan: spatial validation and explanation agreement

Code, Google Earth Engine scripts, flood inventory and results for:

> Ahmad, M., Ullah, Q.Z. *Spatial Validation and Explanation Agreement in Machine-Learning Flood Susceptibility Mapping: A Case Study of the Swat Basin, Pakistan.* Submitted, 2026. Software DOI: `[Zenodo DOI — add after archiving]`

## Contents

| Path | What it is |
|---|---|
| `gee/` | Earth Engine scripts: export of the 21 predictor layers, terrain and hydrology layers, and the per-orbit Sentinel-1 flood labels for 2022 and 2025 |
| `code/` | Python code for preprocessing, models, experiments, explanations and figures (see below) |
| `data/labels/` | Flood inventory GeoTIFFs (30 m, EPSG:32643): combined 2022+2025 and per event. Values: 1 flood, 0 non-flood, 255 unknown. `v1_mixed_orbit/` holds the earlier mixed-orbit version used in the label-sensitivity check |
| `data/ne_*.geojson` | Natural Earth boundaries used in the maps |
| `runs/` | Results. `cv_all_models*/`: repeated spatial CV for all four models with all bands and with SAR, optical or terrain bands removed (per-split test probabilities included). `xai_agreement/`: band importances and rank agreement for five explanation methods. `original_submission/`: results of the original-split experiments |
| `trained_models/` | Trained XGBoost, patch CNN, SegFormer and multi-branch SegFormer from the original split |
| `figures/` | All main and supplementary figures (PDF) |

The 36-band predictor cube (about 6 GB) and the source rasters are not included. They can be regenerated from the Earth Engine scripts; all inputs are open data (Copernicus Sentinel-1/2 and GLO-30, MERIT Hydro, CHIRPS, ESA WorldCover, JRC Global Surface Water).

## Reproducing the results

Run all Python scripts from inside `code/`; they read from `../data` and write to `../runs` and `../figures`.

1. **Predictors**: run `gee/GEE_Swat_Data_Export.js` and `gee/GEE_Swat_Terrain_Features.js` in the Earth Engine Code Editor and place the exported GeoTIFFs in `data/raw/`.
2. **Labels**: use `data/labels/`, or regenerate them with `gee/GEE_Swat_Flood_Labels.js`.
3. **Predictor cube and splits**: `python preprocess_pipeline.py`
4. **Original split**: `python train_baselines.py`, `python train_cnn.py`; SegFormer and multi-branch SegFormer on a GPU with `colab_paste.py` / `colab_multibranch_paste.py` (or `colab_train_segformer.ipynb`).
5. **Repeated spatial CV, all four models**: `python cv_all_models.py`
6. **Data-source tests (retraining without a source)**: `python cv_all_models.py --drop SAR`, `--drop OPT`, `--drop TER`
7. **Explanation agreement on the same models**: `python xai_agreement.py`
8. **Other experiments**: `python experiments_cv_transfer.py transfer` (temporal transfer), `python ablation_terrain.py` (16 vs 36 bands), `python validate_unosat.py` (needs the UNOSAT FL20220808PAK product, available from UNOSAT), `python fourier_sar_analysis.py`
9. **Explanations and figures**: `python xai_analysis.py`, `python dl_xai.py igcam|perm`, `python post_multibranch.py`, `python make_figures.py`, `python make_figures2.py`, `python make_final_map.py`, `python revision_figures.py`, `python make_supp_tables.py`

Random seeds are fixed throughout. Steps 5--7 run on a CPU (roughly one hour each on a 20-thread machine). Package versions are in `requirements.txt`.

## Main results

- Repeated spatial CV (5 block splits, 36 bands): AUC 0.913 (random forest), 0.926 (XGBoost), 0.927 (LightGBM), 0.932 (patch CNN). A single split overstated AUC by 0.018--0.026.
- Without the Sentinel-1 predictors AUC fell by only 0.007--0.013.
- Retraining without terrain variables cost the tree models 0.071--0.081 AUC but the CNN only 0.011.
- Explanation methods agreed within a model (mean Spearman rho 0.82) but not across model types (0.50).

## Licence and citation

Code: MIT (see `LICENSE`). Flood inventory and results: CC BY 4.0; please cite the paper. Source datasets keep their providers' licences. Citation metadata is in `CITATION.cff`.
