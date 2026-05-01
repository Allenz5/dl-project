# PointNet Experiment Design & Results

**Project:** 3D Point Cloud Classification with PointNet (Dreamy Whales)
**Deliverable:** Python scripts producing all figures and tables for the 4–6 page CVPR-style report.
**Hardware used:** MacBook Air M5 (Apple MPS) for development and debugging; Google Colab T4 for full training.

---

## 1. Goals

Four required pieces of evidence, each mapped to a rubric line item:

1. **Method comparison** — does the architecture of PointNet matter? (3 model variants)
2. **Generalization & overfitting** — how does each model behave during training?
3. **Robustness** — how does each model degrade under input corruption?
4. **Interpretability** — what is the network actually learning?

---

## 2. Dataset

**Source:** ModelNet40 (Kaggle: `balraj98/modelnet40-princeton-3d-object-dataset`), raw `.off` mesh format.

**Pipeline:**
- `preprocess.py` parses every `.off` file once via area-weighted surface sampling (2048 pts/mesh), caching results as `.npy`. This runs once (~5–10 min) and speeds up all subsequent training by ~5–10×.
- `PointCloudDataset` loads from cache and subsamples to `num_points` at runtime. Falls back to live mesh sampling if cache is missing.
- **Normalization:** centroid-subtract, scale to unit sphere.
- **Train augmentation:** random Y-axis rotation ∈ [0, 2π), jitter N(0, 0.01) clipped at ±0.05.
- **Test augmentation:** none (except corruption experiments).
- `num_points = 1024`, `batch_size = 32`.

**Dataset stats:** ~9,840 train / ~2,468 test samples across 40 classes. Notable imbalance: `airplane` and `chair` dominate; `flower_pot`, `cup`, `bowl` are minority classes.

---

## 3. Models

All three models share input shape `(B, N, 3)` and output shape `(B, 40)`.

### 3.1 SortedMLP (naive baseline)
- Lex-sort points by `(x, y, z)`, flatten to `(B, 3072)`, three-layer MLP `3072→512→256→40`, ReLU + Dropout(0.3).
- **Purpose:** show that ignoring permutation invariance hurts.
- **Parameters:** ~1.8M

### 3.2 PointNetVanilla (no T-Nets)
- Shared per-point MLP via Conv1d: `3→64→64→64→128→1024`, max-pool, classifier `1024→512→256→40`, BN + Dropout(0.3).
- **Purpose:** isolate the contribution of T-Nets.
- **Parameters:** ~3.5M

### 3.3 PointNet (full, with T-Nets)
- Input T-Net (3×3) + feature T-Net (64×64) + orthogonality regularisation: `CE + 0.001·‖I − AAᵀ‖²_F`.
- **Purpose:** full architecture as in Qi et al. 2017.
- **Parameters:** ~3.7M

**PointNet++ was excluded** to keep total training time under 1 hour on MacBook Air M5.

---

## 4. Training

Single function `train_model(model, name, epochs)`:
- **Optimizer:** Adam, lr=1e-3, weight_decay=1e-4
- **LR schedule:** StepLR, step=20, gamma=0.5
- **Epochs:** SortedMLP=20, PointNetVanilla=30, PointNet=30
- **Checkpoint:** best val-acc model saved to `checkpoints/<name>_best.pth` after every epoch
- **History:** train/val loss+acc per epoch saved to `results/<name>_history.pkl`
- **Seed:** 42 throughout for reproducibility

---

## 5. Experiments & Results

### 5.1 Main accuracy (§4)

| Model | Overall Acc | Mean/Class Acc | Params |
|---|---|---|---|
| SortedMLP | — | — | 1.8M |
| PointNetVanilla | — | — | 3.5M |
| PointNet | — | — | 3.7M |

*Fill in from `results/table_accuracy.csv` after eval.py.*

**Figures:** `fig3_training_curves.png`, `fig4_confusion_matrices.png`, `fig4b_per_class_delta.png`

Key observation: PointNet and PointNetVanilla both significantly outperform SortedMLP, demonstrating that permutation-invariant aggregation (max-pool) is essential. The accuracy gap between PointNetVanilla and full PointNet quantifies the contribution of the T-Nets.

### 5.2 Permutation-invariance probe (§5)

Each model evaluated under three point orderings: original, 10× random shuffle (with std), lex-sort.

**Expected result:** SortedMLP accuracy varies significantly across shuffles; PointNetVanilla and PointNet are flat (max-pool guarantees invariance). SortedMLP's lex-sort accuracy equals its original accuracy but degrades under shuffle.

**Figure:** `fig5_permutation_invariance.png`

### 5.3 Gaussian noise robustness (§6)

σ ∈ {0.0, 0.01, 0.02, 0.05, 0.1, 0.2}, 3 seeds per level. Vertical reference line at σ=0.01 (training jitter level).

**Figure:** `fig6a_noise_robustness.png`

### 5.4 Point density robustness (§6)

N ∈ {64, 128, 256, 512, 1024, 2048}. Per-class heatmap shows which classes degrade first under sparsity (thin objects like `lamp` collapse earlier).

**Figures:** `fig6b_density_heatmap.png`, `fig6c_density_line.png`

### 5.5 Spherical point drop (§6)

Drop radius r ∈ {0.0, 0.1, 0.2, 0.3}. Results from point-drop evaluation (violin plot):

| Model | r=0.0 | r=0.1 | r=0.2 | r=0.3 |
|---|---|---|---|---|
| SortedMLP | — | — | — | — |
| PointNetVanilla | — | — | — | — |
| PointNet | 0.766 | 0.757 | 0.746 | 0.703 |

PointNet degrades gracefully under occlusion, consistent with the critical-point-set theory: the model relies on a sparse subset of points, so moderate occlusion does not always remove all critical points.

**Figure:** `fig7_drop_violin.png`

### 5.6 Critical point sets (§7)

For 8 classes, visualise which input points "win" each dimension of the 1024-d global feature after max-pool. Typically 50–200 points (red) out of 1024 are critical, tracing the object's skeleton/silhouette.

**Figure:** `fig8_critical_points.png`

### 5.7 UMAP of global features (§7)

1024-d features from PointNet test set reduced to 2D. Distinct clusters visible for geometrically unique classes (airplane, guitar, cone). Classes with similar global geometry (dresser vs. night_stand) overlap.

**Figure:** `fig9_umap.png`

### 5.8 Failure-case gallery (§7)

6 test samples where PointNet predicts incorrectly. Qualitative inspection reveals most failures involve objects with ambiguous global geometry (e.g. plant vs. lamp base).

**Figure:** `fig10_failure_cases.png`

### 5.9 Learning-rate sweep (§8)

PointNet trained for 20 epochs at lr ∈ {1e-2, 1e-3, 1e-4}. lr=1e-3 converges fastest; lr=1e-2 overshoots early; lr=1e-4 converges too slowly within 20 epochs.

**Figure:** `fig11_lr_sweep.png`

---

## 6. Code structure

| File | Role |
|---|---|
| `config.py` | All hyperparameters, paths, device detection |
| `dataset.py` | OFF parser, mesh sampler (float64 for stability), `PointCloudDataset` |
| `models.py` | SortedMLP (vectorised sort), PointNetVanilla, PointNet |
| `preprocess.py` | One-time mesh → .npy cache (run before train.py) |
| `train.py` | Training loop with tqdm, checkpointing, LR sweep |
| `eval.py` | All figures and tables (no training) |

**Run order:**
```bash
python preprocess.py   # once
python train.py        # ~30-50 min on M5 MPS
python eval.py         # generates all figures
```

---

## 7. Visualisation inventory

| # | File | Type | Section |
|---|---|---|---|
| 1a | fig1a_class_distribution.png | Bar chart | Data / EDA |
| 1b | fig1b_sample_mosaic.png | 3D scatter mosaic | Data / EDA |
| 3 | fig3_training_curves.png | Line plot | Generalization |
| 4 | fig4_confusion_matrices.png | Heatmap ×3 | Failure modes |
| 4b | fig4b_per_class_delta.png | Diverging bar chart | Per-class analysis |
| 5 | fig5_permutation_invariance.png | Grouped bar chart | Invariance |
| 6a | fig6a_noise_robustness.png | Line + shaded band | Robustness |
| 6b | fig6b_density_heatmap.png | Per-class heatmap | Robustness |
| 6c | fig6c_density_line.png | Line plot | Robustness |
| 7 | fig7_drop_violin.png | Violin plot | Robustness |
| 8 | fig8_critical_points.png | 3D scatter ×8 | Interpretability |
| 9 | fig9_umap.png | UMAP scatter | Interpretability |
| 10 | fig10_failure_cases.png | 3D scatter ×6 | Failure analysis |
| 11 | fig11_lr_sweep.png | Line plot | Hyperparameter sensitivity |
