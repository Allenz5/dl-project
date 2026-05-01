"""
Generate all figures and tables after training.

Usage:
    python eval.py

Requires checkpoints saved by train.py.
Figures → results/fig*.png
Tables  → results/table_accuracy.csv
"""
import json
import pickle
import warnings

import matplotlib
matplotlib.use('Agg')  # no GUI window needed
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
import umap

warnings.filterwarnings('ignore')

from config import (
    DEVICE, DATA_DIR, RESULTS_DIR, CKPT_DIR,
    NUM_POINTS, NUM_CLASSES, BATCH_SIZE,
    LR_STEP, LR_GAMMA, WEIGHT_DECAY, FEAT_REG_WEIGHT,
    SEED, NUM_WORKERS,
)
from dataset import PointCloudDataset, find_dataset_dir
from models import SortedMLP, PointNetVanilla, PointNet

# ── Setup ─────────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}")
dataset_dir = find_dataset_dir(DATA_DIR)
with open(RESULTS_DIR / 'classes.json') as f:
    classes = json.load(f)
CLASS2IDX = {c: i for i, c in enumerate(classes)}
IDX2CLASS  = {i: c for c, i in CLASS2IDX.items()}

test_ds     = PointCloudDataset(dataset_dir, 'test', NUM_POINTS, augment=False, classes=classes)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
print(f"Test: {len(test_ds):,}")

COLORS  = {'SortedMLP': '#e41a1c', 'PointNetVanilla': '#377eb8', 'PointNet': '#4daf4a'}
MARKERS = {'SortedMLP': 'o',       'PointNetVanilla': 's',        'PointNet': '^'}
PALETTE = ['#e41a1c', '#377eb8', '#4daf4a']


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def load_model(model, name):
    path = CKPT_DIR / f'{name}_best.pth'
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


def get_preds(model):
    ps, ls = [], []
    with torch.no_grad():
        for pts, lbl in test_loader:
            out = model(pts.to(DEVICE))
            lg  = out[0] if isinstance(out, tuple) else out
            ps.extend(lg.argmax(1).cpu().numpy())
            ls.extend(lbl.numpy())
    return np.array(ps), np.array(ls)


def per_class_acc(preds, labels, nc=NUM_CLASSES):
    return np.array(
        [(preds[labels == c] == c).mean() if (labels == c).any() else 0.0
         for c in range(nc)]
    )


# ── Load models & histories ───────────────────────────────────────────────
all_models = {
    'SortedMLP':       load_model(SortedMLP(NUM_POINTS, NUM_CLASSES), 'SortedMLP'),
    'PointNetVanilla': load_model(PointNetVanilla(NUM_CLASSES),        'PointNetVanilla'),
    'PointNet':        load_model(PointNet(NUM_CLASSES),               'PointNet'),
}
histories = {}
for name in all_models:
    with open(RESULTS_DIR / f'{name}_history.pkl', 'rb') as f:
        histories[name] = pickle.load(f)

print("\nCollecting predictions...")
all_preds = {}
for name, m in all_models.items():
    all_preds[name], gt_labels = get_preds(m)
    print(f"  {name}: {(all_preds[name] == gt_labels).mean():.4f}")


# ── §1 EDA ────────────────────────────────────────────────────────────────
print("\n[1/11] Class distribution...")
train_ds    = PointCloudDataset(dataset_dir, 'train', NUM_POINTS, augment=False, classes=classes)
tr_cnt      = np.bincount(np.array([l for _, l in train_ds.samples]), minlength=NUM_CLASSES)
te_cnt      = np.bincount(np.array([l for _, l in test_ds.samples]),  minlength=NUM_CLASSES)
order       = np.argsort(tr_cnt)[::-1]

fig, ax = plt.subplots(figsize=(18, 5))
x = np.arange(NUM_CLASSES)
ax.bar(x - .2, tr_cnt[order], .4, label='Train', color='steelblue', alpha=.85)
ax.bar(x + .2, te_cnt[order], .4, label='Test',  color='coral',     alpha=.85)
ax.set_xticks(x)
ax.set_xticklabels([IDX2CLASS[i] for i in order], rotation=90, fontsize=8)
ax.set_ylabel('Samples')
ax.set_title('ModelNet40 — Class Distribution')
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig1a_class_distribution.png', dpi=150)
plt.close()

print("[2/11] Sample mosaic...")
viz_classes = ['airplane', 'chair', 'car', 'guitar', 'lamp', 'piano', 'toilet', 'cone']
viz_idx     = [CLASS2IDX[c] for c in viz_classes]
samples_3d  = {}
for pts, lbl in test_ds:
    if lbl in viz_idx and lbl not in samples_3d:
        samples_3d[lbl] = pts.numpy()
    if len(samples_3d) == len(viz_idx):
        break

fig = plt.figure(figsize=(20, 8))
for i, ci in enumerate(viz_idx):
    ax  = fig.add_subplot(2, 4, i + 1, projection='3d')
    pts = samples_3d[ci]
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=1, c=pts[:, 1], cmap='viridis', alpha=.7)
    ax.set_title(IDX2CLASS[ci], fontsize=10)
    ax.set_axis_off()
plt.suptitle('ModelNet40 — Sample Point Clouds', fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig1b_sample_mosaic.png', dpi=150)
plt.close()


# ── §4 Accuracy table ─────────────────────────────────────────────────────
print("[3/11] Accuracy table + training curves...")
rows = []
for name in all_models:
    p    = all_preds[name]
    oa   = accuracy_score(gt_labels, p)
    mpca = per_class_acc(p, gt_labels).mean()
    h    = histories[name]
    rows.append({
        'Model':       name,
        'Overall Acc': f'{oa:.4f}',
        'Mean/Class':  f'{mpca:.4f}',
        'Params(M)':   f'{count_params(all_models[name]) / 1e6:.2f}',
        'Train h':     f'{sum(h["epoch_time"]) / 3600:.2f}',
    })
acc_df = pd.DataFrame(rows)
print(acc_df.to_string(index=False))
acc_df.to_csv(RESULTS_DIR / 'table_accuracy.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, h in histories.items():
    c  = COLORS[name]
    ep = range(1, len(h['train_acc']) + 1)
    axes[0].plot(ep, h['train_acc'],  c=c, ls='-',  lw=1.5, label=name)
    axes[0].plot(ep, h['val_acc'],    c=c, ls='--', lw=1.5, alpha=.7)
    axes[1].plot(ep, h['train_loss'], c=c, ls='-',  lw=1.5, label=name)
    axes[1].plot(ep, h['val_loss'],   c=c, ls='--', lw=1.5, alpha=.7)
for ax, lbl in zip(axes, ['Accuracy', 'Loss']):
    ax.set_xlabel('Epoch')
    ax.set_ylabel(lbl)
    ax.set_title(f'Train (—) / Val (- -) {lbl}')
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig3_training_curves.png', dpi=150)
plt.close()

print("[4/11] Confusion matrices...")
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
tick_locs = list(range(0, NUM_CLASSES, 5))
for ax, (name, p) in zip(axes, all_preds.items()):
    cm = confusion_matrix(gt_labels, p, normalize='true')
    im = ax.imshow(cm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(tick_locs)
    ax.set_yticks(tick_locs)
    ax.set_xticklabels([IDX2CLASS[i] for i in tick_locs], rotation=90, fontsize=7)
    ax.set_yticklabels([IDX2CLASS[i] for i in tick_locs], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
plt.suptitle('Normalised Confusion Matrices', fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig4_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

print("[5/11] Per-class delta...")
pca              = {name: per_class_acc(all_preds[name], gt_labels) for name in all_models}
delta_naive      = pca['PointNet'] - pca['SortedMLP']
delta_vanilla    = pca['PointNet'] - pca['PointNetVanilla']
s_idx            = np.argsort(delta_naive)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, delta, title in [
    (axes[0], delta_naive[s_idx],   'PointNet vs SortedMLP (Δ acc)'),
    (axes[1], delta_vanilla[s_idx], 'PointNet vs PointNetVanilla (Δ acc)'),
]:
    clrs = ['#d73027' if v < 0 else '#4575b4' for v in delta]
    ax.barh([IDX2CLASS[i] for i in s_idx], delta, color=clrs, alpha=.85)
    ax.axvline(0, color='k', lw=.8)
    ax.set_xlabel('Δ Accuracy')
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='y', labelsize=7)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig4b_per_class_delta.png', dpi=150, bbox_inches='tight')
plt.close()


# ── §5 Permutation invariance ─────────────────────────────────────────────
print("\n[6/11] Permutation invariance probe...")

def eval_order(model, ordering='original', n_trials=1):
    accs = []
    for t in range(n_trials):
        rng     = np.random.RandomState(SEED + t)
        correct = total = 0
        with torch.no_grad():
            for pts, lbl in DataLoader(test_ds, 32, shuffle=False, num_workers=NUM_WORKERS):
                if ordering == 'shuffle':
                    for b in range(pts.size(0)):
                        pts[b] = pts[b][torch.from_numpy(rng.permutation(pts.size(1)))]
                elif ordering == 'lex':
                    for b in range(pts.size(0)):
                        k = pts[b, :, 0] * 1e8 + pts[b, :, 1] * 1e4 + pts[b, :, 2]
                        pts[b] = pts[b][k.argsort()]
                lg = model(pts.to(DEVICE))
                if isinstance(lg, tuple):
                    lg = lg[0]
                correct += (lg.argmax(1).cpu() == lbl).sum().item()
                total   += lbl.size(0)
        accs.append(correct / total)
    return accs

N_SHUF   = 10
perm_res = {}
for name, m in all_models.items():
    orig = eval_order(m, 'original')[0]
    shuf = eval_order(m, 'shuffle', N_SHUF)
    lex  = eval_order(m, 'lex')[0]
    perm_res[name] = {'orig': orig, 'shuf_mean': np.mean(shuf),
                      'shuf_std': np.std(shuf), 'lex': lex}
    print(f"  {name}  orig={orig:.4f}  shuf={np.mean(shuf):.4f}±{np.std(shuf):.4f}  lex={lex:.4f}")

names = list(perm_res.keys())
x = np.arange(len(names))
w = .25
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w, [perm_res[n]['orig']      for n in names], w, label='Original',            color='#1f77b4', alpha=.85)
ax.bar(x,     [perm_res[n]['shuf_mean'] for n in names], w, label=f'Shuffle (×{N_SHUF})', color='#ff7f0e', alpha=.85,
       yerr=[perm_res[n]['shuf_std'] for n in names], capsize=4)
ax.bar(x + w, [perm_res[n]['lex']       for n in names], w, label='Lex-sort',             color='#2ca02c', alpha=.85)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel('Test Accuracy')
ax.set_ylim(0, 1)
ax.set_title('Permutation-Invariance Probe')
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig5_permutation_invariance.png', dpi=150)
plt.close()


# ── §6 Noise robustness ───────────────────────────────────────────────────
print("\n[7/11] Gaussian noise robustness...")
NOISE_SIGMAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]

def eval_noise(model, sigma, n_seeds=3):
    accs = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed + 100)
        c = t = 0
        with torch.no_grad():
            for pts, lbl in DataLoader(test_ds, 32, shuffle=False, num_workers=NUM_WORKERS):
                if sigma > 0:
                    pts = pts + torch.from_numpy(
                        rng.normal(0, sigma, pts.shape).astype(np.float32))
                lg = model(pts.to(DEVICE))
                if isinstance(lg, tuple):
                    lg = lg[0]
                c += (lg.argmax(1).cpu() == lbl).sum().item()
                t += lbl.size(0)
        accs.append(c / t)
    return accs

noise_res = {name: [] for name in all_models}
for name, m in all_models.items():
    for s in NOISE_SIGMAS:
        accs = eval_noise(m, s)
        noise_res[name].append({'s': s, 'mean': np.mean(accs), 'std': np.std(accs)})
    print("  " + name + ": " + "  ".join(f"σ={r['s']:.2f}→{r['mean']:.3f}" for r in noise_res[name]))

fig, ax = plt.subplots(figsize=(10, 6))
for name, res in noise_res.items():
    c  = COLORS[name]
    xs = [r['s'] for r in res]
    ys = [r['mean'] for r in res]
    es = [r['std']  for r in res]
    ax.plot(xs, ys, c=c, marker=MARKERS[name], lw=2, label=name)
    ax.fill_between(xs, [y - e for y, e in zip(ys, es)],
                        [y + e for y, e in zip(ys, es)], color=c, alpha=.15)
ax.axvline(0.01, color='gray', ls=':', lw=1.5, label='Train jitter σ')
ax.set_xlabel('Gaussian noise σ')
ax.set_ylabel('Test Accuracy')
ax.set_title('Robustness to Gaussian Noise')
ax.legend()
ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig6a_noise_robustness.png', dpi=150)
plt.close()

print("[8/11] Point density...")
DENSITY_LEVELS = [64, 128, 256, 512, 1024, 2048]

def eval_density(model, model_name, n_pts):
    ds = PointCloudDataset(dataset_dir, 'test', n_pts, augment=False, classes=classes)
    ps, ls = [], []
    with torch.no_grad():
        for pts, lbl in DataLoader(ds, 32, shuffle=False, num_workers=NUM_WORKERS):
            if model_name == 'SortedMLP':
                B, N, C = pts.shape
                tgt = NUM_POINTS
                if N < tgt:
                    pts = torch.cat([pts, torch.zeros(B, tgt - N, C)], 1)
                else:
                    pts = pts[:, :tgt]
            lg = model(pts.to(DEVICE))
            if isinstance(lg, tuple):
                lg = lg[0]
            ps.extend(lg.argmax(1).cpu().numpy())
            ls.extend(lbl.numpy())
    return np.array(ps), np.array(ls)

density_res = {}
for name, m in all_models.items():
    density_res[name] = {}
    for n in DENSITY_LEVELS:
        p, l = eval_density(m, name, n)
        density_res[name][n] = {'preds': p, 'labels': l, 'acc': (p == l).mean()}
    print("  " + name + ": " + "  ".join(
        f"N={n}→{density_res[name][n]['acc']:.3f}" for n in DENSITY_LEVELS))

fig, axes = plt.subplots(1, 3, figsize=(24, 7))
for ax, name in zip(axes, all_models):
    mat = np.array([per_class_acc(density_res[name][n]['preds'], density_res[name][n]['labels'])
                    for n in DENSITY_LEVELS])
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([IDX2CLASS[i] for i in range(NUM_CLASSES)], rotation=90, fontsize=6)
    ax.set_yticks(range(len(DENSITY_LEVELS)))
    ax.set_yticklabels([str(n) for n in DENSITY_LEVELS])
    ax.set_ylabel('# Points')
    ax.set_title(f'{name} — Per-class acc vs density')
    plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig6b_density_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
for name in all_models:
    ax.plot(DENSITY_LEVELS, [density_res[name][n]['acc'] for n in DENSITY_LEVELS],
            marker='o', c=COLORS[name], label=name)
ax.set_xscale('log')
ax.set_xlabel('# Points')
ax.set_ylabel('Test Accuracy')
ax.set_title('Accuracy vs Point Density')
ax.legend()
ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig6c_density_line.png', dpi=150)
plt.close()

print("[9/11] Point drop (violin)...")
DROP_RADII = [0.0, 0.1, 0.2, 0.3]

def sphere_drop(pts, r, rng):
    if r == 0:
        return pts
    c    = pts[rng.randint(len(pts))]
    keep = np.linalg.norm(pts - c, axis=1) > r
    rem  = pts[keep]
    return rem if len(rem) >= 64 else pts

rng_drop       = np.random.RandomState(42)
all_pts_raw    = [pts.numpy() for pts, _ in test_ds]
gt_labels_drop = np.array([lbl for _, lbl in test_ds.samples])
dropped_clouds = {r: [sphere_drop(p, r, rng_drop) for p in all_pts_raw] for r in DROP_RADII}

def eval_drop(model, dropped_list, n_pts=NUM_POINTS):
    rng = np.random.RandomState(42)
    accs, batch_pts, batch_lbl = [], [], []

    def run_batch(bpts, blbl):
        t = torch.tensor(np.stack(bpts), dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            lg = model(t)
            if isinstance(lg, tuple):
                lg = lg[0]
        return (lg.argmax(1).cpu().numpy() == np.array(blbl)).tolist()

    for i, (d, lbl) in enumerate(zip(dropped_list, gt_labels_drop)):
        idx = rng.choice(len(d), n_pts, replace=(len(d) < n_pts))
        batch_pts.append(d[idx])
        batch_lbl.append(lbl)
        if len(batch_pts) == 64 or i == len(dropped_list) - 1:
            accs.extend(run_batch(batch_pts, batch_lbl))
            batch_pts, batch_lbl = [], []
    return np.array(accs)

drop_res = {name: {} for name in all_models}
for name, m in all_models.items():
    for r in DROP_RADII:
        drop_res[name][r] = eval_drop(m, dropped_clouds[r])
    print("  " + name + ": " + "  ".join(
        f"r={r}→{drop_res[name][r].mean():.3f}" for r in DROP_RADII))

n_m, n_r = len(all_models), len(DROP_RADII)
base = np.arange(n_r) * (n_m + 1.5)
fig, ax = plt.subplots(figsize=(12, 6))
for mi, (name, color) in enumerate(zip(all_models, PALETTE)):
    pos   = base + mi
    data  = [drop_res[name][r].astype(float) for r in DROP_RADII]
    parts = ax.violinplot(data, positions=pos, widths=.8, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(.6)
    for k in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
        if k in parts:
            parts[k].set_color(color)
ax.set_xticks(base + (n_m - 1) / 2)
ax.set_xticklabels([f'r={r}' for r in DROP_RADII])
ax.set_ylabel('Per-sample accuracy')
ax.set_title('Accuracy Distribution Under Spherical Point Drop')
ax.legend(handles=[mpatches.Patch(color=c, label=n) for n, c in zip(all_models, PALETTE)])
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig7_drop_violin.png', dpi=150)
plt.close()


# ── §7 Interpretability ───────────────────────────────────────────────────
print("\n[10/11] Critical point sets + UMAP + failure cases...")
pn_model = all_models['PointNet']

def critical_mask(model, pts_t):
    x = pts_t.to(DEVICE).transpose(1, 2)
    with torch.no_grad():
        x = torch.bmm(model.t3(x), x)
        x = model.e1(x)
        A = model.t64(x)
        x = torch.bmm(A, x)
        x = model.e2(x)
    _, idx = x.max(2)
    mask = torch.zeros(pts_t.size(1), dtype=torch.bool)
    mask[torch.unique(idx.cpu())] = True
    return mask.numpy()

viz_cls    = ['airplane', 'chair', 'car', 'guitar', 'lamp', 'piano', 'toilet', 'cone']
viz_ci     = [CLASS2IDX[c] for c in viz_cls]
crit_samps = {}
for pts, lbl in test_ds:
    if lbl in viz_ci and lbl not in crit_samps:
        crit_samps[lbl] = pts
    if len(crit_samps) == len(viz_ci):
        break

fig = plt.figure(figsize=(20, 8))
for i, ci in enumerate(viz_ci):
    pts_t = crit_samps[ci].unsqueeze(0)
    cm    = critical_mask(pn_model, pts_t)
    pts_n = pts_t[0].numpy()
    ax    = fig.add_subplot(2, 4, i + 1, projection='3d')
    ax.scatter(pts_n[~cm, 0], pts_n[~cm, 2], pts_n[~cm, 1], s=1, c='lightgray', alpha=.4)
    ax.scatter(pts_n[cm,  0], pts_n[cm,  2], pts_n[cm,  1], s=8, c='red',       alpha=.9)
    ax.set_title(f'{IDX2CLASS[ci]}\n({cm.sum()} critical)', fontsize=9)
    ax.set_axis_off()
plt.suptitle('PointNet — Critical Point Sets (red)', fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig8_critical_points.png', dpi=150)
plt.close()

print("  UMAP (this takes ~1 min)...")
feats, labs = [], []
with torch.no_grad():
    for pts, lbl in test_loader:
        _, f, _ = pn_model(pts.to(DEVICE), return_feat=True)
        feats.append(f.cpu().numpy())
        labs.extend(lbl.numpy())
f_pn  = np.vstack(feats)
l_pn  = np.array(labs)
emb   = umap.UMAP(n_components=2, random_state=SEED).fit_transform(f_pn)

sel_cls = ['airplane', 'chair', 'car', 'guitar', 'lamp', 'piano', 'toilet', 'cone', 'sofa', 'table']
sel_idx = [CLASS2IDX[c] for c in sel_cls]
cmap10  = plt.cm.get_cmap('tab10', 10)
fig, ax = plt.subplots(figsize=(10, 8))
for ci, cls_i in enumerate(sel_idx):
    mask = l_pn == cls_i
    ax.scatter(emb[mask, 0], emb[mask, 1], s=5, color=cmap10(ci),
               label=IDX2CLASS[cls_i], alpha=.7)
ax.set_title('PointNet — Global Feature Space (UMAP)', fontsize=12)
ax.set_xlabel('UMAP-1')
ax.set_ylabel('UMAP-2')
ax.legend(fontsize=8, markerscale=2, ncol=2)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig9_umap.png', dpi=150, bbox_inches='tight')
plt.close()

pn_p   = all_preds['PointNet']
fail_i = np.where(pn_p != gt_labels)[0]
gallery = []
for idx in fail_i:
    pts, lbl = test_ds[idx]
    gallery.append((pts.numpy(), IDX2CLASS[lbl], IDX2CLASS[pn_p[idx]]))
    if len(gallery) == 6:
        break
fig = plt.figure(figsize=(18, 7))
for i, (pts, true_c, pred_c) in enumerate(gallery):
    ax = fig.add_subplot(2, 3, i + 1, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=2, c=pts[:, 1], cmap='plasma', alpha=.7)
    ax.set_title(f'True: {true_c}\nPred: {pred_c}', fontsize=9)
    ax.set_axis_off()
plt.suptitle('PointNet — Failure Cases', fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig10_failure_cases.png', dpi=150)
plt.close()


# ── §8 LR sweep figure ────────────────────────────────────────────────────
print("\n[11/11] LR sweep figure...")
lr_hist_path = RESULTS_DIR / 'lr_history.pkl'
if lr_hist_path.exists():
    with open(lr_hist_path, 'rb') as f:
        lr_hist = pickle.load(f)
    LR_COLORS = {1e-2: '#d62728', 1e-3: '#1f77b4', 1e-4: '#2ca02c'}
    fig, ax = plt.subplots(figsize=(10, 5))
    for lr_val, va in lr_hist.items():
        ax.plot(range(1, len(va) + 1), va, c=LR_COLORS[lr_val], lw=2, label=f'lr={lr_val:.0e}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy')
    ax.set_title('PointNet — Learning Rate Sensitivity (30 epochs)')
    ax.legend()
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'fig11_lr_sweep.png', dpi=150)
    plt.close()
else:
    print("  lr_history.pkl not found — re-run train.py without --skip-lr")


# ── Summary ───────────────────────────────────────────────────────────────
print("\n=== Done ===")
print(acc_df.to_string(index=False))
print(f"\nFigures saved to: {RESULTS_DIR}")
FIGS = [
    'fig1a_class_distribution', 'fig1b_sample_mosaic',
    'fig3_training_curves', 'fig4_confusion_matrices', 'fig4b_per_class_delta',
    'fig5_permutation_invariance', 'fig6a_noise_robustness',
    'fig6b_density_heatmap', 'fig6c_density_line', 'fig7_drop_violin',
    'fig8_critical_points', 'fig9_umap', 'fig10_failure_cases', 'fig11_lr_sweep',
]
for f in FIGS:
    p = RESULTS_DIR / f'{f}.png'
    print(f"  {'ok' if p.exists() else 'MISSING':6s}  {f}.png")
