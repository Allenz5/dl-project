import torch
from pathlib import Path

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
CKPT_DIR    = BASE_DIR / 'checkpoints'

for d in [RESULTS_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else
    'cpu'
)

NUM_POINTS      = 1024
NUM_CLASSES     = 40
BATCH_SIZE      = 32
LR              = 1e-3
LR_STEP         = 20
LR_GAMMA        = 0.5
WEIGHT_DECAY    = 1e-4
FEAT_REG_WEIGHT = 0.001
JITTER_SIGMA    = 0.01
JITTER_CLIP     = 0.05
SEED            = 42
NUM_WORKERS     = 0
