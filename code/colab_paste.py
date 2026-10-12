# ================= PASTE INTO ONE COLAB CELL AND RUN =================
# Needs Drive folder "swat_flood" with: segformer_flood.py,
# feature_cube.npy, valid_mask.npy, flood_labels.tif
# Runtime -> Change runtime type -> T4 GPU

# --- 1. Mount Google Drive and check the uploads ---------------------
from google.colab import drive
drive.mount('/content/drive')

import os, shutil
SRC = '/content/drive/MyDrive/swat_flood'
for f in ['segformer_flood.py', 'feature_cube.npy',
          'valid_mask.npy', 'flood_labels.tif']:
    assert os.path.exists(f'{SRC}/{f}'), f'MISSING in Drive: {f}'
print('All 4 files found in Drive')

# --- 2. Install the one missing package, confirm the GPU -------------
!pip -q install rasterio
import torch
print('CUDA available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'No GPU! Runtime -> Change runtime type -> T4 GPU'

# --- 3. Copy data to fast local disk (training off Drive is slow) ----
os.makedirs('/content/work/data/processed', exist_ok=True)
os.makedirs('/content/work/data/labels', exist_ok=True)
shutil.copy(f'{SRC}/segformer_flood.py', '/content/work/')
shutil.copy(f'{SRC}/feature_cube.npy',  '/content/work/data/processed/')
shutil.copy(f'{SRC}/valid_mask.npy',    '/content/work/data/processed/')
shutil.copy(f'{SRC}/flood_labels.tif',  '/content/work/data/labels/')
os.chdir('/content/work')
print('Data copied to local disk')

# --- 4. Import the training module, point it at the local files ------
import sys; sys.path.insert(0, '/content/work')
import segformer_flood as sf
sf.CFG.update({
    'cube_path':  'data/processed/feature_cube.npy',
    'valid_path': 'data/processed/valid_mask.npy',
    'label_path': 'data/labels/flood_labels.tif',
    'out_dir':    'runs/segformer',
    'num_workers': 2,
})

# --- 5. Train (early stopping ends it well before 100 epochs) --------
model, cube, valid = sf.train(sf.CFG)

# --- 6. Full-basin susceptibility map from the best checkpoint -------
prob = sf.predict_map(model, cube, valid, sf.CFG)

# --- 7. Copy results back to Drive -----------------------------------
os.makedirs(f'{SRC}/results', exist_ok=True)
for f in ['best_model.pt', 'history.json', 'susceptibility.npy']:
    shutil.copy(f'runs/segformer/{f}', f'{SRC}/results/')
print('DONE - results saved to Drive: swat_flood/results/')
# ======================================================================
