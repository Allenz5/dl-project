"""
Pre-sample all ModelNet40 meshes once → .npy files in data/cache/
Run this ONCE before train.py. Takes ~5-10 min, saves hours of training.

    python preprocess.py
"""
import numpy as np
from tqdm import tqdm
from config import DATA_DIR, NUM_POINTS
from dataset import find_dataset_dir, read_off, sample_mesh

CACHE_POINTS = 2048          # cache at 2048 so density experiments can subsample
CACHE_DIR    = DATA_DIR / 'cache'

dataset_dir = find_dataset_dir(DATA_DIR)
off_files   = sorted(dataset_dir.glob('**/*.off'))
print(f"Found {len(off_files):,} meshes  →  caching {CACHE_POINTS} pts each to {CACHE_DIR}")

errors = 0
skipped = 0
for path in tqdm(off_files, ncols=80):
    cache = CACHE_DIR / path.relative_to(dataset_dir).with_suffix('.npy')
    if cache.exists():
        skipped += 1
        continue
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        verts, faces = read_off(path)
        pts = sample_mesh(verts, faces, CACHE_POINTS)
        np.save(cache, pts)
    except Exception as e:
        tqdm.write(f"ERROR {path.name}: {e}")
        errors += 1

print(f"Done. cached={len(off_files)-skipped-errors}  skipped={skipped}  errors={errors}")
print("Now run:  python train.py")
