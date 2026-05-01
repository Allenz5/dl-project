"""
Train all models and run the LR sweep.

Usage:
    python train.py              # train all three models + LR sweep
    python train.py --skip-lr    # skip LR sweep (faster)

Checkpoints → checkpoints/<name>_best.pth
Histories   → results/<name>_history.pkl
LR sweep    → results/lr_history.pkl
"""
import argparse
import json
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEVICE, DATA_DIR, RESULTS_DIR, CKPT_DIR,
    NUM_POINTS, NUM_CLASSES, BATCH_SIZE,
    LR, LR_STEP, LR_GAMMA, WEIGHT_DECAY, FEAT_REG_WEIGHT,
    SEED, NUM_WORKERS,
)
from dataset import PointCloudDataset, find_dataset_dir
from models import SortedMLP, PointNetVanilla, PointNet


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def make_loaders(dataset_dir, classes):
    train_ds = PointCloudDataset(dataset_dir, 'train', NUM_POINTS, augment=True,  classes=classes)
    test_ds  = PointCloudDataset(dataset_dir, 'test',  NUM_POINTS, augment=False, classes=classes)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
    return train_loader, test_loader, train_ds, test_ds


def train_model(model, name, epochs, train_loader, test_loader):
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, LR_STEP, LR_GAMMA)
    crit  = nn.CrossEntropyLoss()
    is_pn = isinstance(model, PointNet)

    hist = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'epoch_time']}
    best_acc  = 0.0
    ckpt_path = CKPT_DIR / f'{name}_best.pth'

    print(f"\n{'='*60}")
    print(f"  Training {name}  ({epochs} epochs)  device={DEVICE}")
    print(f"{'='*60}")

    for ep in range(1, epochs + 1):
        t0 = time.time()

        # train
        model.train()
        tl, tc, tn = 0.0, 0, 0
        bar = tqdm(train_loader, desc=f"[{name}] ep {ep:3d}/{epochs} train",
                   leave=False, ncols=90, unit='batch')
        for pts, lbl in bar:
            pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
            opt.zero_grad()
            if is_pn:
                logits, _, A = model(pts, return_feat=True)
                loss = crit(logits, lbl) + FEAT_REG_WEIGHT * PointNet.feat_reg(A)
            else:
                logits = model(pts)
                loss   = crit(logits, lbl)
            loss.backward()
            opt.step()
            tl += loss.item() * pts.size(0)
            tc += (logits.argmax(1) == lbl).sum().item()
            tn += pts.size(0)
            bar.set_postfix(loss=f'{tl/tn:.4f}', acc=f'{tc/tn:.3f}')
        bar.close()

        # val
        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for pts, lbl in test_loader:
                pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
                lg = model(pts)
                if isinstance(lg, tuple):
                    lg = lg[0]
                vl += crit(lg, lbl).item() * pts.size(0)
                vc += (lg.argmax(1) == lbl).sum().item()
                vn += pts.size(0)

        ta, va   = tc / tn, vc / vn
        tl_, vl_ = tl / tn, vl / vn
        et = time.time() - t0
        for k, v in zip(['train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'epoch_time'],
                        [tl_, ta, vl_, va, opt.param_groups[0]['lr'], et]):
            hist[k].append(v)

        ckpt_flag = ''
        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), ckpt_path)
            ckpt_flag = '  ← best'

        sched.step()
        print(f"[{name}] ep {ep:3d}/{epochs}  "
              f"train {ta:.4f} / loss {tl_:.4f}  "
              f"val {va:.4f} / loss {vl_:.4f}  "
              f"{et:.1f}s{ckpt_flag}")

    with open(RESULTS_DIR / f'{name}_history.pkl', 'wb') as f:
        pickle.dump(hist, f)
    print(f"\n[{name}] done — best val acc = {best_acc:.4f}  ckpt → {ckpt_path}\n")
    return hist


def lr_sweep(train_loader, test_loader):
    lr_values = [1e-2, 1e-3, 1e-4]
    lr_hist   = {}
    crit      = nn.CrossEntropyLoss()

    print(f"\n{'='*60}")
    print(f"  LR sweep on PointNet (30 epochs × 3 learning rates)")
    print(f"{'='*60}")

    for lr_val in lr_values:
        print(f"\n--- lr = {lr_val:.0e} ---")
        set_seed()
        m   = PointNet(NUM_CLASSES).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=lr_val, weight_decay=WEIGHT_DECAY)
        sch = torch.optim.lr_scheduler.StepLR(opt, LR_STEP, LR_GAMMA)
        va_hist = []

        for ep in tqdm(range(1, 21), desc=f'lr={lr_val:.0e}', ncols=60):
            m.train()
            for pts, lbl in train_loader:
                pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
                opt.zero_grad()
                logits, _, A = m(pts, return_feat=True)
                loss = crit(logits, lbl) + FEAT_REG_WEIGHT * PointNet.feat_reg(A)
                loss.backward()
                opt.step()
            sch.step()

            m.eval()
            c = t = 0
            with torch.no_grad():
                for pts, lbl in test_loader:
                    pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
                    c += (m(pts).argmax(1) == lbl).sum().item()
                    t += lbl.size(0)
            va_hist.append(c / t)

        print(f"  final val acc = {va_hist[-1]:.4f}")
        lr_hist[lr_val] = va_hist

    with open(RESULTS_DIR / 'lr_history.pkl', 'wb') as f:
        pickle.dump(lr_hist, f)
    print(f"\nLR sweep done → {RESULTS_DIR / 'lr_history.pkl'}")
    return lr_hist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-lr', action='store_true', help='skip LR sweep')
    args = parser.parse_args()

    set_seed()
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    dataset_dir = find_dataset_dir(DATA_DIR)
    classes     = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    print(f"Dataset: {dataset_dir}  ({len(classes)} classes)")

    train_loader, test_loader, train_ds, test_ds = make_loaders(dataset_dir, classes)
    print(f"Train: {len(train_ds):,}  Test: {len(test_ds):,}")

    # save class list for eval.py
    with open(RESULTS_DIR / 'classes.json', 'w') as f:
        json.dump(classes, f)

    train_model(SortedMLP(NUM_POINTS, NUM_CLASSES), 'SortedMLP',       20, train_loader, test_loader)
    train_model(PointNetVanilla(NUM_CLASSES),        'PointNetVanilla', 30, train_loader, test_loader)
    train_model(PointNet(NUM_CLASSES),               'PointNet',        30, train_loader, test_loader)

    if not args.skip_lr:
        lr_sweep(train_loader, test_loader)

    print("\nAll done. Run:  python eval.py")
