# ====== RESUME: EVALUATE RESCUED MULTI-BRANCH CHECKPOINT (one Colab cell) ======
# For the day after an interrupted training run. Requires in Drive/swat_flood:
#   segformer_flood.py, multibranch_flood.py, feature_cube.npy (36-band, 6 GB),
#   valid_mask.npy, flood_labels.tif, metadata.json,
#   results_multibranch/best_model.pt   <- the rescued checkpoint
# Runtime -> Change runtime type -> T4 GPU. Takes ~15 min total.

from google.colab import drive
drive.mount('/content/drive')

import os, shutil, json
SRC = '/content/drive/MyDrive/swat_flood'
need = ['segformer_flood.py', 'multibranch_flood.py', 'feature_cube.npy',
        'valid_mask.npy', 'flood_labels.tif', 'metadata.json',
        'results_multibranch/best_model.pt']
for f in need:
    assert os.path.exists(f'{SRC}/{f}'), f'MISSING in Drive: {f}'
sz = os.path.getsize(f'{SRC}/feature_cube.npy') / 1e9
assert sz > 4, f'cube is {sz:.1f} GB — old 16-band version, re-upload!'
print('All files found')

!pip -q install rasterio
import torch
assert torch.cuda.is_available(), 'No GPU! Runtime -> Change runtime type -> T4'

os.makedirs('/content/work/data/processed', exist_ok=True)
os.makedirs('/content/work/data/labels', exist_ok=True)
os.makedirs('/content/work/runs/multibranch', exist_ok=True)
shutil.copy(f'{SRC}/segformer_flood.py',  '/content/work/')
shutil.copy(f'{SRC}/multibranch_flood.py','/content/work/')
shutil.copy(f'{SRC}/feature_cube.npy',    '/content/work/data/processed/')
shutil.copy(f'{SRC}/metadata.json',       '/content/work/data/processed/')
shutil.copy(f'{SRC}/valid_mask.npy',      '/content/work/data/processed/')
shutil.copy(f'{SRC}/flood_labels.tif',    '/content/work/data/labels/')
shutil.copy(f'{SRC}/results_multibranch/best_model.pt',
            '/content/work/runs/multibranch/')
os.chdir('/content/work')
print('Data staged')

import sys; sys.path.insert(0, '/content/work')
import numpy as np
import segformer_flood as sf
sf.CFG.update({'cube_path': 'data/processed/feature_cube.npy',
               'valid_path': 'data/processed/valid_mask.npy',
               'label_path': 'data/labels/flood_labels.tif',
               'num_workers': 2})
import multibranch_flood as mb
from torch.utils.data import DataLoader

# ---- rebuild the SAME test split (seeded identically to training) ----------
groups, names = mb.band_groups('data/processed/metadata.json')
cube = np.load(sf.CFG['cube_path'])                      # GPU box: load fully
valid = np.load(sf.CFG['valid_path'])
labels = sf.load_labels(sf.CFG['label_path'], valid.shape)
rng = np.random.default_rng(sf.CFG['seed'])
splits = sf.build_tiles(valid, labels, sf.CFG, rng)
ds = sf.FloodTileDataset(cube, labels, valid, splits['test'],
                         sf.CFG['tile'], augment=False, noise_std=0)
test_dl = DataLoader(ds, batch_size=sf.CFG['batch_size'], shuffle=False,
                     num_workers=2, pin_memory=True)
print('test tiles:', len(ds))

# ---- load rescued checkpoint, evaluate, run branch ablation ----------------
state = torch.load('runs/multibranch/best_model.pt',
                   map_location=mb.DEVICE, weights_only=False)
model = mb.MultiBranchSegFormer(state['groups'],
                                state['branch_cfg']).to(mb.DEVICE)
model.load_state_dict(state['model']); model.eval()
print('checkpoint: best epoch', state['epoch'],
      '| val:', {k: round(v, 4) for k, v in state['val_metrics'].items()})

test = sf.evaluate(model, test_dl)
print('TEST:', {k: round(v, 4) for k, v in test.items()})
print('Branch ablation (sensor-level attribution):')
ablation = mb.branch_ablation(model, test_dl, state['groups'])

with open('runs/multibranch/history.json', 'w') as f:
    json.dump({'history': [], 'test': test, 'ablation': ablation,
               'groups': {g: len(i) for g, i in state['groups'].items()},
               'best_epoch': state['epoch'],
               'best_val': state['val_metrics'],
               'note': 'training interrupted; eval from rescued checkpoint'},
              f, indent=2)
shutil.copy('runs/multibranch/history.json', f'{SRC}/results_multibranch/')
print('DONE - history.json in Drive: swat_flood/results_multibranch/')
# =============================================================================
