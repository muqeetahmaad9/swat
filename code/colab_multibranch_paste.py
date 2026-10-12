# ============== MULTI-BRANCH RUN — PASTE INTO ONE COLAB CELL ==============
# Drive folder "swat_flood" must ALSO contain (new uploads, NO renaming):
#   feature_cube.npy      <- data/processed/feature_cube.npy (6 GB, 36-band —
#                            REPLACES the old 2.7 GB one when Drive asks)
#   metadata.json         <- data/processed/metadata.json
#   multibranch_flood.py  <- code/multibranch_flood.py
# (segformer_flood.py, valid_mask.npy, flood_labels.tif are already there)
# Runtime -> Change runtime type -> T4 GPU

from google.colab import drive
drive.mount('/content/drive')

import os, shutil
SRC = '/content/drive/MyDrive/swat_flood'
need = ['segformer_flood.py', 'multibranch_flood.py', 'feature_cube.npy',
        'valid_mask.npy', 'flood_labels.tif', 'metadata.json']
for f in need:
    assert os.path.exists(f'{SRC}/{f}'), f'MISSING in Drive: {f}'
# guard: the 36-band cube is ~6 GB; the old 16-band one was ~2.7 GB.
sz = os.path.getsize(f'{SRC}/feature_cube.npy') / 1e9
assert sz > 4, (f'feature_cube.npy in Drive is {sz:.1f} GB — still the OLD '
                '16-band cube. Upload the new 6 GB one and choose Replace.')
print('All 6 files found; cube is the 36-band version')

!pip -q install rasterio
import torch
assert torch.cuda.is_available(), 'No GPU! Runtime -> Change runtime type -> T4 GPU'

os.makedirs('/content/work/data/processed', exist_ok=True)
os.makedirs('/content/work/data/labels', exist_ok=True)
shutil.copy(f'{SRC}/segformer_flood.py',  '/content/work/')
shutil.copy(f'{SRC}/multibranch_flood.py','/content/work/')
shutil.copy(f'{SRC}/feature_cube.npy',    '/content/work/data/processed/')
shutil.copy(f'{SRC}/metadata.json',       '/content/work/data/processed/')
shutil.copy(f'{SRC}/valid_mask.npy',      '/content/work/data/processed/')
shutil.copy(f'{SRC}/flood_labels.tif',    '/content/work/data/labels/')
os.chdir('/content/work')
print('Data staged')

import sys; sys.path.insert(0, '/content/work')
import segformer_flood as sf
sf.CFG.update({'cube_path': 'data/processed/feature_cube.npy',
               'valid_path': 'data/processed/valid_mask.npy',
               'label_path': 'data/labels/flood_labels.tif',
               'num_workers': 2})

import multibranch_flood as mb
mb.train()          # trains, evaluates, runs the branch ablation, saves results

os.makedirs(f'{SRC}/results_multibranch', exist_ok=True)
for f in ['best_model.pt', 'history.json']:
    shutil.copy(f'runs/multibranch/{f}', f'{SRC}/results_multibranch/')
print('DONE - results in Drive: swat_flood/results_multibranch/')
# ===========================================================================
