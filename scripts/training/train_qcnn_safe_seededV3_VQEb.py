#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_qcnn_safe_seededV3_patched.py
QCNN • Safe Seeded V3 — patch pour reproductibilité + cache angles (CNN->qubits)
Usage: python train_qcnn_safe_seededV3_patched.py --cache_angles True
"""

import os
import time
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import pennylane as qml

# ---------------------------------------------------------------------
# Defaults : adapte si tu veux
# ---------------------------------------------------------------------
DEFAULT_TRAIN = r"C:\ARKEOCERA\arkeocera\IA_ARKEO\ceramique\crops_dataset_equilibre\train"
DEFAULT_VAL   = r"C:\ARKEOCERA\arkeocera\IA_ARKEO\ceramique\crops_dataset_equilibre\val"
DEFAULT_OUT   = r"C:\ARKEOCERA\arkeocera\IA_ARKEO\ceramique\runs_qcnn_seededV3"
# ---------------------------------------------------------------------

def set_seed_all(seed=42, nthreads=4):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(nthreads)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"[INFO] seed={seed} applied, threads={nthreads}")

# BalancedBatchSampler (renvoie des batches d'indices)
class BalancedBatchSampler:
    def __init__(self, labels, batch_size, drop_last=True, resample_short=False, seed=42):
        self.labels = list(map(int, labels))
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.resample_short = bool(resample_short)
        self.rng = random.Random(seed)
        self.class_to_indices = defaultdict(list)
        for idx, y in enumerate(self.labels):
            self.class_to_indices[y].append(idx)
        self.classes = sorted(self.class_to_indices.keys())
        self.num_classes = len(self.classes)
        if self.num_classes <= 1:
            raise ValueError("BalancedBatchSampler requires >=2 classes.")
        if self.batch_size < self.num_classes:
            raise ValueError("batch_size < number of classes.")
        self.samples_per_class = self.batch_size // self.num_classes
        self.remainder = self.batch_size - self.samples_per_class * self.num_classes

    def __iter__(self):
        # create shuffled pools for each class
        pools = {}
        ptrs = {}
        for c in self.classes:
            perm = list(self.class_to_indices[c])
            self.rng.shuffle(perm)
            pools[c] = perm
            ptrs[c] = 0
        while True:
            batch = []
            for c in self.classes:
                need = self.samples_per_class
                for _ in range(need):
                    if ptrs[c] >= len(pools[c]):
                        if self.resample_short:
                            self.rng.shuffle(pools[c]); ptrs[c] = 0
                        else:
                            if not self.drop_last and batch:
                                yield batch
                            return
                    batch.append(pools[c][ptrs[c]])
                    ptrs[c] += 1
            if self.remainder > 0:
                ci = 0
                for _ in range(self.remainder):
                    c = self.classes[ci % self.num_classes]
                    if ptrs[c] >= len(pools[c]):
                        if self.resample_short:
                            self.rng.shuffle(pools[c]); ptrs[c] = 0
                        else:
                            if not self.drop_last and batch:
                                yield batch
                            return
                    batch.append(pools[c][ptrs[c]])
                    ptrs[c] += 1
                    ci += 1
            if len(batch) == self.batch_size:
                yield batch
            else:
                if not self.drop_last and batch:
                    yield batch
                return

    def __len__(self):
        minc = min(len(v) for v in self.class_to_indices.values())
        k = max(1, self.samples_per_class)
        return max(1, minc // k)

# ---------------------------------------------------------------------
# Simple feature extractor (CNN) — on garde léger et déterministe
# ---------------------------------------------------------------------
class SmallCNNFeat(nn.Module):
    def __init__(self, in_channels=3, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = feat_dim
        self.reduce = nn.Linear(128, feat_dim)

    def forward(self, x):
        x = self.net(x)            # (B,128,1,1)
        x = x.view(x.size(0), -1)  # (B,128)
        x = self.reduce(x)         # (B,feat_dim)
        return x

# ---------------------------------------------------------------------
# Quantum layer (AngleEmbedding + entangling + returns Z expectation per qubit)
# ---------------------------------------------------------------------
def make_qnode(n_qubits, dev_name="default.qubit"):
    dev = qml.device(dev_name, wires=n_qubits, shots=None)

    @qml.qnode(dev, interface='torch')
    def qnode(angles):
        # angles: tensor shape (n_qubits,) or (batch, n_qubits) not batched here
        qml.templates.AngleEmbedding(angles, wires=range(n_qubits))
        # simple entangler
        qml.templates.StronglyEntanglingLayers(weights=torch.zeros((1, n_qubits, 3)), wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return qnode

# wrapper to run qnode per-sample (vectorization optional later)
def qnode_batch_forward(qnode, angles_batch, device):
    # angles_batch: torch.Tensor (B, n_qubits)
    outs = []
    for i in range(angles_batch.size(0)):
        a = angles_batch[i].detach().cpu().numpy().astype(float)
        res = qnode(a)  # returns list or tensor
        res_t = torch.tensor(res, dtype=torch.float32)
        outs.append(res_t)
    return torch.stack(outs, dim=0).to(device)  # (B, n_qubits)

# ---------------------------------------------------------------------
# Compute angles (CNN features -> to_qubits mapping) and cache
# ---------------------------------------------------------------------
def compute_angles_and_cache(feature_extractor, dataloader, device, to_qubits_fn, outpath):
    feature_extractor.eval()
    angles_all = []
    labels_all = []
    with torch.no_grad():
        for imgs, labs in tqdm(dataloader, desc="Compute angles", leave=False):
            imgs = imgs.to(device)
            feats = feature_extractor(imgs)   # (B, feat_dim)
            # normalize and map to [-pi,pi]
            ang = torch.tanh(to_qubits_fn(feats)) * math.pi
            angles_all.append(ang.cpu().numpy())
            labels_all.append(labs.numpy())
    angles = np.vstack(angles_all)
    labels = np.concatenate(labels_all)
    np.savez_compressed(outpath, angles=angles, labels=labels)
    print(f"[INFO] Cached angles saved to {outpath}.npz (shape {angles.shape})")
    return outpath + ".npz"

# ---------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------
def evaluate(model_clf, dataloader, device):
    model_clf.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model_clf(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)
    return f1, acc, report

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default=DEFAULT_TRAIN)
    parser.add_argument("--val_dir", default=DEFAULT_VAL)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache_angles", type=lambda x: x.lower() in ("1","true","yes"), default=True)
    parser.add_argument("--n_qubits", type=int, default=8)
    args = parser.parse_args()

    set_seed_all(args.seed, nthreads=4)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # Transforms & datasets
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    train_ds = ImageFolder(args.train_dir, transform=transform)
    val_ds   = ImageFolder(args.val_dir, transform=transform)
    class_names = train_ds.classes
    n_classes = len(class_names)
    print(f"[INFO] classes={class_names}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Feature extractor + to_qubits mapping
    feat_dim = 128
    feature_extractor = SmallCNNFeat(in_channels=3, feat_dim=feat_dim).to(device)
    to_qubits = nn.Linear(feat_dim, args.n_qubits).to(device)

    # Quantum qnode (PennyLane)
    qnode = make_qnode(args.n_qubits, dev_name="default.qubit")

    # Classical classifier: take qnode outputs (n_qubits) -> FC -> logits
    clf = nn.Sequential(
        nn.Linear(args.n_qubits, 64),
        nn.ReLU(),
        nn.Linear(64, n_classes)
    ).to(device)

    # optimizer over classical params + to_qubits + feat reduce
    optimizer = optim.AdamW(list(feature_extractor.parameters()) +
                            list(to_qubits.parameters()) +
                            list(clf.parameters()), lr=0.002, weight_decay=0.0005)

    criterion = nn.CrossEntropyLoss()

    # Dataloaders for computing angles (no Balanced sampler needed here)
    dl_train_for_cache = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    dl_val_for_cache   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

    # cache paths
    cache_train_path = str(out_dir / "angles_train")
    cache_val_path   = str(out_dir / "angles_val")

    # compute or load angles
    if args.cache_angles:
        # compute or load train
        npz_train = out_dir / "angles_train.npz"
        npz_val   = out_dir / "angles_val.npz"
        if npz_train.exists() and npz_val.exists():
            print("[INFO] Loading cached angles (train & val)")
            d_train = np.load(str(npz_train)); angles_train = d_train["angles"]; labels_train = d_train["labels"]
            d_val   = np.load(str(npz_val));   angles_val   = d_val["angles"];   labels_val   = d_val["labels"]
        else:
            print("[INFO] Computing angles and caching (cela peut prendre quelques minutes)...")
            compute_angles_and_cache(feature_extractor, dl_train_for_cache, device, to_qubits, cache_train_path)
            compute_angles_and_cache(feature_extractor, dl_val_for_cache, device, to_qubits, cache_val_path)
            d_train = np.load(str(npz_train)); angles_train = d_train["angles"]; labels_train = d_train["labels"]
            d_val   = np.load(str(npz_val));   angles_val   = d_val["angles"];   labels_val   = d_val["labels"]
        # TensorDatasets
        X_train = torch.tensor(angles_train, dtype=torch.float32)
        y_train = torch.tensor(labels_train, dtype=torch.long)
        X_val   = torch.tensor(angles_val, dtype=torch.float32)
        y_val   = torch.tensor(labels_val, dtype=torch.long)
        ds_train = TensorDataset(X_train, y_train)
        ds_val   = TensorDataset(X_val, y_val)
        # Balanced sampler based on labels
        sampler_train = BalancedBatchSampler(y_train.numpy().tolist(), batch_size=args.batch, seed=args.seed)
        train_loader = DataLoader(ds_train, batch_sampler=sampler_train, num_workers=0, pin_memory=False)
        val_loader   = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)
        print("[INFO] Using cached-angles dataloaders (fast per epoch).")
    else:
        # No caching: use image dataloaders + Balanced sampler (slower)
        labels_train_list = [y for _, y in train_ds.samples]
        sampler_train = BalancedBatchSampler(labels_train_list, batch_size=args.batch, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_sampler=sampler_train, num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)
        print("[INFO] Using direct-image dataloaders (slower).")

    # ---- training loop ----
    best_f1 = -1.0
    best_path = out_dir / "best_qcnn.pt"
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        feature_extractor.train()
        to_qubits.train()
        clf.train()
        running_loss = 0.0
        it = 0
        for batch in train_loader:
            if args.cache_angles:
                angles_batch, labels = batch
                labels = labels.to(device)
                # angles are already "angles" in radians shape (B, n_qubits)
                angles_batch = angles_batch.to(device)
            else:
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                feats = feature_extractor(imgs)
                angles_batch = torch.tanh(to_qubits(feats)) * math.pi  # (B, n_qubits)

            # quantum forward (per-sample)
            qouts = qnode_batch_forward(qnode, angles_batch, device)  # (B, n_qubits)
            logits = clf(qouts)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            it += 1

        # eval
        f1, acc, report = evaluate(lambda x: clf(qnode_batch_forward(qnode, x.to(device), device)), val_loader, device) if args.cache_angles else evaluate(lambda x: clf(qnode_batch_forward(qnode, x.to(device), device)), val_loader, device)

        epoch_time = time.time() - epoch_t0
        print(f"[E{epoch}/{args.epochs}] loss={running_loss/it:.4f} f1={f1:.4f} acc={acc:.4f} time={epoch_time:.1f}s")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "feature_extractor": feature_extractor.state_dict(),
                "to_qubits": to_qubits.state_dict(),
                "clf": clf.state_dict(),
                "epoch": epoch,
                "best_f1": best_f1
            }, str(best_path))
            print(f"[INFO] nouveau best -> {best_path} (f1={best_f1:.4f})")

    total_time = time.time() - start
    print(f"[DONE] training terminé. total time: {total_time:.1f}s. best_f1={best_f1:.4f}")
    print(f"[INFO] best model saved at {best_path}")

if __name__ == "__main__":
    main()
