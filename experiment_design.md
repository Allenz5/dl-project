# PointNet Experiment Design Document

**Project:** 3D Point Cloud Classification with PointNet (Dreamy Whales)
**Deliverable:** A single Jupyter notebook (`pointnet_experiments.ipynb`) producing every figure, table, and number needed for the 4–6 page CVPR-style report.
**Target environment:** Google Colab with GPU (T4 / L4).

---

## 1. Goals

The notebook is engineered backwards from the rubric. Every experiment must produce evidence that maps to a specific rubric line item. The four required pieces of evidence are:

1. **Method comparison** — does the design of PointNet matter? (4 model variants)
2. **Generalization & overfitting** — how does each model behave during training?
3. **Robustness** — how does each model degrade under input corruption?
4. **Interpretability** — what is the network actually learning?

We deliberately use **different visualizations for each** so that the report is not a wall of line plots.

---

## 2. Notebook structure (sections = cells groups)

| § | Section | Output / Artifact |
|---|---|---|
| 0 | Setup, seeds, config dict, GPU check | Reproducibility prelude |
| 1 | ModelNet40 download + dataset class + EDA | Class-distribution bar, 3D class-mosaic, datasheet block |
| 2 | Four model definitions | Architecture summary tables, parameter counts |
| 3 | Shared training loop + train all 4 | Loss/accuracy histories saved as pickles |
| 4 | Quantitative evaluation | Accuracy table, training curves, confusion matrices |
| 5 | Permutation-invariance probe | Bar chart with shuffle-variance error bars |
| 6 | Robustness suite (noise, density, drop) | Line plots **+ per-class heatmaps** |
| 7 | Interpretation suite | **Critical-point-set** renders, **t-SNE/UMAP** of global features, failure-case gallery |
| 8 | Hyperparameter sensitivity (mini) | Learning-rate sweep on full PointNet |
| 9 | Save figures + summary table for the report | `/results/` directory |

The notebook is one file but each `§` is its own cell group with markdown headers, so cells can be re-run independently.

---

## 3. Data (§1)

**Source:** ModelNet40 (Kaggle: `balraj98/modelnet40-princeton-3d-object-dataset`). Use the Princeton-style `_normal_resampled` variant if available — pre-sampled 10k points per object means we don't need to mesh-sample.

**Pipeline:**

- `PointCloudDataset(split, num_points, augment)`
  - Loads cached `.npy`/`.txt` per object, samples `num_points` uniformly without replacement.
  - **Normalization:** centroid-subtract, scale to unit sphere. Justified in report.
  - **Train augmentation:** random rotation around Y-axis ∈ [0, 2π), jitter `N(0, 0.01)` clipped at ±0.05. Toggleable.
  - **Test augmentation:** none, except for the corruption experiments.
- `train_loader`: batch 32, shuffle, num_workers 2.
- `test_loader`: batch 32, no shuffle.

**Defaults:** `num_points = 1024` for training (matches the original paper).

**EDA cells (figures for the report):**

- Bar chart of class counts (train vs test) — flags `airplane` and `chair` as dominant; flags `flower_pot`, `cup`, `bowl` as minority. Establishes class-imbalance context for *why* mean-per-class accuracy matters.
- 3D scatter mosaic: one example per class for 8 visually distinct classes (`airplane`, `chair`, `car`, `guitar`, `lamp`, `piano`, `toilet`, `cone`). Uses `matplotlib` `Axes3D`. This is the report's "what the data looks like" figure.
- Histogram of points-per-object (sanity that all are ≥1024).

**Datasheet block** (markdown, ~6 bullets) covering: motivation, composition, collection, preprocessing, intended uses, biases. Maps directly to the rubric's "Datasheets for Datasets" requirement.

---

## 4. Models (§2)

All four models share the same input shape `(B, N, 3)` and output shape `(B, 40)`. Implementing them in one notebook lets us isolate each architectural choice as an ablation.

### 4.1 Naive baseline — `SortedMLP`

- Lex-sort points by `(x, y, z)`, flatten to `(B, 3·N)`.
- Three-layer MLP: `3072 → 512 → 256 → 40`, ReLU + Dropout(0.3).
- **Purpose:** establish that ignoring permutation invariance hurts, even with sorting.
- **Expected result:** ~30–50% accuracy, very high variance under shuffle.

### 4.2 PointNet without T-Net — `PointNetVanilla`

- Shared per-point MLP: `3 → 64 → 64 → 64 → 128 → 1024` (1×1 conv).
- Max-pool over the N dimension → 1024-d global feature.
- Classifier MLP: `1024 → 512 → 256 → 40`, BN + Dropout(0.3).
- **Purpose:** isolate the contribution of T-Nets.
- **Expected result:** ~85% accuracy, fully permutation-invariant.

### 4.3 Full PointNet — `PointNet`

- Input T-Net: small PointNet that outputs a `3×3` matrix; multiply input by it.
- Shared MLP `3 → 64`.
- Feature T-Net: outputs `64×64` matrix; multiply features by it.
- Shared MLP `64 → 128 → 1024`, max-pool, classifier.
- **Loss:** `CE + 0.001 · ||I − A Aᵀ||_F²` (orthogonality reg on feature transform).
- **Expected result:** ~88–89% accuracy, slight improvement on rotated test inputs.

### 4.4 PointNet++ (SSG) — `PointNetPP`

- Two Set Abstraction layers (single-scale grouping):
  - SA1: FPS sample 512 centroids, ball-query radius 0.2, k=32, MLP [64, 64, 128].
  - SA2: FPS sample 128 centroids, radius 0.4, k=64, MLP [128, 128, 256].
- Global SA: MLP [256, 512, 1024], aggregate to single vector.
- Classifier head identical to PointNet's.
- **Purpose:** does hierarchical local feature aggregation help on noisy / sparse inputs?
- **Expected result:** ~90–91% accuracy, much more robust under density loss.

**Implementation note:** to keep the notebook self-contained for Colab, include a minimal pure-PyTorch FPS + ball-query implementation rather than depending on `pointnet2_ops` (which needs a CUDA build).

**Architecture summary cell:** print parameter count and approximate FLOPs for each model in a small `pandas` table.

---

## 5. Training (§3)

A single function `train_model(model, name, epochs)`:

- Optimizer: **Adam**, lr `1e-3`, weight decay `1e-4`.
- LR schedule: **StepLR**, step 20, gamma 0.5.
- Loss: cross-entropy (+ feature-reg term for full PointNet).
- Logged per epoch: train loss, train acc, val loss, val acc, lr, epoch time.
- Best model (by val acc) checkpointed to Drive.
- Histories saved to `results/{name}_history.pkl`.

**Training schedule decisions:**

| Model | Epochs | Justification |
|---|---|---|
| SortedMLP | 30 | Plateaus quickly; longer just overfits |
| PointNetVanilla | 50 | Standard schedule from the paper |
| PointNet | 50 | Same |
| PointNetPP | 60 | Slightly slower convergence due to grouping |

All four trained back-to-back in a single notebook run (~1.5–2 hours on a Colab L4).

**Reproducibility:** single seed (42) propagated to torch, numpy, random; `torch.backends.cudnn.deterministic = True`.

---

## 6. Evaluation (§4)

| Output | Type | Report use |
|---|---|---|
| Overall accuracy + mean-per-class accuracy table | `pandas.DataFrame` | Main results table |
| Train/val loss + accuracy curves | 2×2 grid, models overlaid per panel | Generalization figure ("did it overfit?") |
| Confusion matrices | 2×2 grid of normalized heatmaps | Failure-mode figure |
| Per-class accuracy delta (PN vs Naive, PN++ vs PN) | Diverging bar chart | "Where did the architecture help?" |
| Param count + train time + val acc | Single tidy table | Cost/benefit narrative |

The 2×2 confusion-matrix grid is one of the highest-value figures because it shows *which classes confuse each model*, not just aggregate numbers.

---

## 7. Permutation-invariance probe (§5)

For each model, run test-set inference under three orderings:

1. Original ordering (as loaded).
2. Random shuffle, 10 independent trials.
3. Lexicographic sort.

Plot the three accuracies as a grouped bar chart, with std-error bars on the shuffle group. **Expected result:** SortedMLP's bars vary wildly across shuffles; the three PointNet variants are visually flat. This is the strongest single piece of evidence that the symmetric max-pool actually delivers permutation invariance.

---

## 8. Robustness suite (§6) — three experiments, three viz styles

Each robustness experiment uses a *different* primary visualization to avoid a wall of line charts.

### 8.1 Gaussian noise injection (line plot)

- `σ ∈ {0.0, 0.01, 0.02, 0.05, 0.1, 0.2}`, applied to test points only.
- Line plot, 4 lines (models), markers, shaded error band over 3 noise seeds.
- Annotate "training jitter level" on the x-axis as a vertical reference line — this contextualizes whether the model is being tested in-distribution or extrapolating.

### 8.2 Point density (per-class heatmap — primary viz)

- `N ∈ {64, 128, 256, 512, 1024, 2048}`.
- For each model, build a heatmap: rows = density, cols = the 40 classes, cells = accuracy.
- This is more informative than a line plot because it shows *which classes lose accuracy first* — e.g., thin objects like `lamp` collapse first under sparsity.
- Aggregate line plot also shown for the report's main figure.

### 8.3 Point drop / occlusion (violin plot)

- Drop a contiguous spherical region of `r ∈ {0.0, 0.1, 0.2, 0.3}` from each cloud.
- Per-sample accuracy → violin plot (one violin per drop level per model).
- Captures distribution, not just mean — communicates that some samples survive aggressive occlusion while others don't.

---

## 9. Interpretation (§7)

### 9.1 Critical point set

For the full PointNet model, after the max-pool, every dimension of the 1024-d global feature is "won" by one of the N input points. The set of unique winners (typically 50–200 points out of 1024) is the **critical point set** — Qi et al.'s key visualization.

- 8 examples covering 8 classes.
- 3D scatter: gray = original points, **red** = critical points.
- Pair with the upper-bound shape (the largest cloud whose critical set is identical) when feasible.

This is the single most "PointNet-y" figure in the paper and signals that we understood the architecture, not just trained it.

### 9.2 t-SNE / UMAP of global features

- Extract 1024-d features from the test set for full PointNet **and** PointNet++.
- Reduce with UMAP (preferred over t-SNE for cluster preservation).
- Side-by-side scatter, colored by class (10 selected classes for legibility, or all 40 with a discrete colormap).
- Discussion point: do PointNet++ features cluster more tightly? This visually confirms the quantitative accuracy difference.

### 9.3 Failure-case gallery

- Find test samples where PointNet was wrong but PointNet++ was right.
- Render 6 of them as 3D scatters with `(true_label, pn_pred, pnpp_pred)` captions.
- Hand-pick examples that illustrate the "local geometry helps" hypothesis.

---

## 10. Hyperparameter sensitivity (§8 — short)

- Train full PointNet with `lr ∈ {1e-2, 1e-3, 1e-4}` for 30 epochs each.
- Overlay validation-accuracy curves.
- One small figure + a paragraph in the report. Establishes that we tuned, not just used defaults.

---

## 11. Hyperparameter & config block

```python
CONFIG = {
    'num_points':      1024,
    'num_classes':     40,
    'batch_size':      32,
    'epochs_default':  50,
    'lr':              1e-3,
    'lr_step':         20,
    'lr_gamma':        0.5,
    'optimizer':       'Adam',
    'weight_decay':    1e-4,
    'feat_reg_weight': 0.001,
    'jitter_sigma':    0.01,
    'jitter_clip':     0.05,
    'seed':            42,
    'num_workers':     2,
    'device':          'cuda',
}
```

Single source of truth — every cell reads from this dict.

---

## 12. Mapping to the 4-page report

| Report section | Notebook source | Key figure(s) |
|---|---|---|
| Introduction / motivation | §1 EDA | Fig 1: ModelNet40 sample mosaic + class distribution |
| Approach | §2 model defs | Fig 2: architecture diagram (drawn separately, not in notebook), Table 1: param counts |
| Experiments — setup | §3 training | Hyperparameter table |
| Experiments — main results | §4 | Fig 3: training curves; Fig 4: confusion matrix grid; Table 2: accuracy |
| Experiments — invariance | §5 | Fig 5: permutation bar chart |
| Experiments — robustness | §6 | Fig 6: noise curves + per-class heatmap; Fig 7: density heatmap |
| Analysis / interpretation | §7 | Fig 8: critical point sets; Fig 9: UMAP |
| Discussion | §6.3, §7.3, §8 | Failure cases, LR sweep |

This produces ~9 figures and ~3 tables — comfortably enough for a 4-page CVPR-style report without padding.

---

## 13. What "different methods and visualizations" means here

The rubric explicitly rewards variety. The notebook deliberately uses:

**Methods (4):** sorted-MLP, vanilla PointNet, full PointNet (T-Nets), PointNet++.

**Visualizations (8 distinct styles):**

1. Bar chart (class distribution, permutation invariance).
2. 3D scatter mosaic (data inspection, critical points, failure cases).
3. Line plot with error bands (noise robustness, training curves).
4. Confusion-matrix heatmap (per-model failure modes).
5. **Per-class density heatmap** (which classes lose first).
6. Diverging bar chart (per-class accuracy delta between models).
7. Violin plot (point-drop distribution).
8. **UMAP scatter** (learned feature space).

No two experiments share a primary visualization style.

---

## 14. Risk register

| Risk | Mitigation |
|---|---|
| ModelNet40 download flaky on Colab | Mirror the `.zip` to Drive on first run; subsequent runs skip download. |
| PointNet++ FPS/grouping slow without CUDA ops | Pure-PyTorch fallback; reduce `num_points` to 1024 (sufficient). |
| Total training time exceeds Colab session | Train one model per session, save checkpoints; or use Colab Pro. |
| t-SNE / UMAP unstable across runs | Fix seed; report both for one model as a sanity sample. |
| Class imbalance dominates results | Always report mean-per-class alongside overall accuracy. |

---

## 15. Open questions for you before I build the notebook

1. **PointNet++ scope:** confirmed in scope per your earlier choice (full plan). I'll include a pure-PyTorch implementation; OK?
2. **Train length:** are you OK with ~2 hours of total training on Colab L4? If you want it shorter, I can drop `num_points` to 512 and shrink epochs; expected accuracy hit is ~1–2 points across the board.
3. **Code reuse:** the proposal cites `fxia22/pointnet.pytorch`. Do you want me to import its modules directly (cleaner, fewer LOC) or reimplement from scratch in the notebook (clearer to a reader, more LOC)? I lean toward **reimplement** because the report rubric asks "what existing code did you start with" and a clean self-contained notebook is easier to defend.
4. **Drive layout:** I'll write outputs to `/content/drive/MyDrive/cs7643_pointnet/` by default — confirm or change.

Once you answer (or just say "go"), I'll generate the notebook end-to-end in the workspace folder.
