import glob
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

from config import DATA_DIR, NUM_POINTS, JITTER_SIGMA, JITTER_CLIP

CACHE_DIR    = DATA_DIR / 'cache'
CACHE_POINTS = 2048


def find_dataset_dir(root=DATA_DIR):
    matches = glob.glob(str(root) + '/**/airplane/train', recursive=True)
    matches = [m for m in matches if 'cache' not in m]
    if matches:
        return Path(matches[0]).parent.parent
    raise FileNotFoundError(
        f"ModelNet40 not found under {root}\n"
        "Expected: data/.../ModelNet40/<class>/train/*.off"
    )


def read_off(path):
    with open(path) as f:
        header = f.readline().strip()
        if header == 'OFF':
            counts = f.readline().strip().split()
        else:
            counts = header[3:].strip().split()
        n_v, n_f = int(counts[0]), int(counts[1])
        verts = np.array([f.readline().split()[:3] for _ in range(n_v)], dtype=np.float32)
        faces = np.array([f.readline().split()[1:4] for _ in range(n_f)], dtype=np.int32)
    return verts, faces


def sample_mesh(verts, faces, n_pts):
    # float64 prevents overflow when vertex coords are large (float32 cross^2 overflows)
    v0 = verts[faces[:, 0]].astype(np.float64)
    v1 = verts[faces[:, 1]].astype(np.float64)
    v2 = verts[faces[:, 2]].astype(np.float64)
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.maximum(
        np.nan_to_num(np.sqrt((cross ** 2).sum(1)) / 2, nan=0.0, posinf=0.0), 1e-10
    )
    probs = areas / areas.sum()
    probs /= probs.sum()  # renormalise: np.random.choice requires exact sum=1
    idx = np.random.choice(len(faces), n_pts, p=probs)
    r1  = np.random.rand(n_pts, 1)
    r2  = np.random.rand(n_pts, 1)
    bad = (r1 + r2) > 1
    r1[bad], r2[bad] = 1 - r1[bad], 1 - r2[bad]
    return (verts[faces[idx, 0]]
            + r1 * (verts[faces[idx, 1]] - verts[faces[idx, 0]])
            + r2 * (verts[faces[idx, 2]] - verts[faces[idx, 0]])).astype(np.float32)


class PointCloudDataset(Dataset):
    def __init__(self, root, split='train', num_points=NUM_POINTS, augment=True, classes=None):
        self.root       = Path(root)
        self.split      = split
        self.num_points = num_points
        self.augment    = augment and (split == 'train')
        self.classes    = classes or sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class2idx  = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            folder = self.root / cls / split
            if not folder.exists():
                continue
            for f in sorted(folder.glob('*.off')):
                self.samples.append((f, self.class2idx[cls]))

    def _normalize(self, pts):
        pts = pts - pts.mean(axis=0)
        return pts / (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)

    def _augment(self, pts):
        theta = np.random.uniform(0, 2 * np.pi)
        c, s  = np.cos(theta), np.sin(theta)
        R     = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        pts   = pts @ R.T
        jitter = np.clip(np.random.normal(0, JITTER_SIGMA, pts.shape),
                         -JITTER_CLIP, JITTER_CLIP)
        return (pts + jitter).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # cache: data/cache/<class>/<split>/<name>.npy  (relative to dataset root)
        cache = CACHE_DIR / path.relative_to(self.root).with_suffix('.npy')
        if cache.exists() and self.num_points <= CACHE_POINTS:
            full   = np.load(cache)
            chosen = np.random.choice(len(full), self.num_points, replace=False)
            pts    = full[chosen]
        else:
            verts, faces = read_off(path)
            pts = sample_mesh(verts, faces, self.num_points)
        pts = self._normalize(pts)
        if self.augment:
            pts = self._augment(pts)
        return torch.tensor(pts, dtype=torch.float32), label
