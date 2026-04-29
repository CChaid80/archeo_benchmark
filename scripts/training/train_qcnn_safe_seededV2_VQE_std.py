#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""train_qcnn_vqe_seeded_std.py

QCNN (CNN -> PQC) variante VQE • entraînement reproductible + sorties standardisées (paper-ready).

Objectif:
- même interface que train_qcnn_seeded_std.py (dataset_id/scenario/split_id/runs_root/test_dir…)
- arborescence normalisée + metrics.json via experiment_utils.py
- mapping canonique des classes stable (0=sigillee, 1=CO, 2=CR), tolère accents/majuscules
- sauvegarde du meilleur checkpoint sur val (macro-F1) puis évaluation finale optionnelle sur test
- export : metrics.csv + reports + confusions

Note:
- La variante "VQE" ici correspond à une tête quantique qui ajoute une observable d'énergie <H>
  (Hamiltonien somme des PauliZ) comme feature supplémentaire, puis une tête linéaire.
"""

import argparse
import csv
import json
import math
import os
import random
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm

import pennylane as qml

from experiment_utils import make_run_dir, canonical_metrics_schema, save_metrics_json, write_json


# ----------------------------
# Reproductibilité
# ----------------------------
def set_seed_all(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        cudnn.deterministic = True
        cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass


# ----------------------------
# Canonical class mapping
# ----------------------------
CANONICAL_NAMES = ["sigillee", "CO", "CR"]
CANONICAL_NORM_TO_IDX = {"sigillee": 0, "co": 1, "cr": 2}


def _strip_accents_lower(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def remap_imagefolder_to_canonical(ds: ImageFolder) -> None:
    # Build mapping orig_idx -> canonical_idx using normalized folder names
    orig_classes = list(ds.classes)
    orig_to_canon: Dict[int, int] = {}
    used = set()
    for orig_idx, name in enumerate(orig_classes):
        norm = _strip_accents_lower(name)
        if norm not in CANONICAL_NORM_TO_IDX:
            raise ValueError(
                f"Classe inconnue '{name}' (norm='{norm}'). Attendu: {list(CANONICAL_NORM_TO_IDX.keys())}"
            )
        canon = CANONICAL_NORM_TO_IDX[norm]
        if canon in used:
            raise ValueError(f"Collision de mapping: plusieurs classes mappent sur l'index canonique {canon}")
        used.add(canon)
        orig_to_canon[orig_idx] = canon

    # Remap samples/targets
    for i, (p, y) in enumerate(ds.samples):
        cy = orig_to_canon[int(y)]
        ds.samples[i] = (p, cy)
    ds.targets = [orig_to_canon[int(y)] for y in ds.targets]

    # Force canonical dict
    ds.classes = CANONICAL_NAMES
    ds.class_to_idx = {c: i for i, c in enumerate(CANONICAL_NAMES)}


# ----------------------------
# BalancedBatchSampler
# ----------------------------
class BalancedBatchSampler(Sampler[List[int]]):
    """Batches équilibrés entre classes. Tolère batch_size non multiple du nb de classes en complétant."""

    def __init__(self, labels, batch_size: int, drop_last: bool = True, resample_short: bool = False, seed: int = 42):
        super().__init__(data_source=None)
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
            raise ValueError("BalancedBatchSampler nécessite >=2 classes.")
        if self.batch_size < self.num_classes:
            raise ValueError("batch_size < nb_classes")

        self.samples_per_class = self.batch_size // self.num_classes
        self.remainder = self.batch_size - self.samples_per_class * self.num_classes

    def __iter__(self):
        pools, ptrs = {}, {}
        for c in self.classes:
            perm = list(self.class_to_indices[c])
            self.rng.shuffle(perm)
            pools[c] = perm
            ptrs[c] = 0

        while True:
            batch = []
            for c in self.classes:
                for _ in range(self.samples_per_class):
                    if ptrs[c] >= len(pools[c]):
                        if self.resample_short:
                            self.rng.shuffle(pools[c])
                            ptrs[c] = 0
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
                            self.rng.shuffle(pools[c])
                            ptrs[c] = 0
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


# ----------------------------
# Loss
# ----------------------------
class FocalSmoothLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, eps: float = 0.01, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        self.eps = float(eps)
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.register_buffer("weight", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        C = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.eps / C)
            true_dist.scatter_(1, targets.unsqueeze(1), 1 - self.eps + self.eps / C)
        ce = -(true_dist * log_probs).sum(dim=1)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


# ----------------------------
# Model: CNN -> (angles) -> PQC(VQE-features) -> Linear head
# ----------------------------
class QuantumLayerVQE(nn.Module):
    """Renvoie [<Z0>.. <Z_{n-1}>, <H>] puis une head linéaire vers num_classes."""

    def __init__(self, n_qubits: int, n_layers: int, num_classes: int, device_name: str = "default.qubit", shots=None):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.num_classes = int(num_classes)

        self.qdev = qml.device(device_name, wires=self.n_qubits, shots=shots)

        # variational parameters
        self.qparams = nn.Parameter(torch.zeros(self.n_layers, self.n_qubits, 2, dtype=torch.float64))

        # simple Hamiltonian (sum Z_i)
        coeffs = [1.0] * self.n_qubits
        obs = [qml.PauliZ(i) for i in range(self.n_qubits)]
        self.hamiltonian = qml.Hamiltonian(coeffs, obs)

        @qml.qnode(self.qdev, interface="torch", diff_method="adjoint")
        def circuit(inputs, weights):
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)
                qml.RY(inputs[w], wires=w)
            for l in range(self.n_layers):
                for w in range(self.n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % self.n_qubits])
                for w in range(self.n_qubits):
                    qml.RZ(weights[l, w, 0], wires=w)
                    qml.RY(weights[l, w, 1], wires=w)
            z_exps = [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]
            energy = qml.expval(self.hamiltonian)
            return z_exps + [energy]

        self.circuit = circuit
        self.head = nn.Linear(self.n_qubits + 1, self.num_classes)

    def forward(self, x_angles: torch.Tensor) -> torch.Tensor:
        outs = []
        B = x_angles.shape[0]
        for i in range(B):
            x_cpu = x_angles[i].to("cpu", dtype=torch.float64)  # keep graph for qml/torch
            out_list = self.circuit(x_cpu, self.qparams)
            out_t = torch.stack(list(out_list), dim=0) if isinstance(out_list, (list, tuple)) else out_list
            out_t = out_t.to(device=x_angles.device, dtype=x_angles.dtype)
            outs.append(out_t.view(-1))
        feats = torch.stack(outs, dim=0)
        return self.head(feats)


class HybridQCNNVQE(nn.Module):
    def __init__(self, num_classes: int, n_qubits: int = 6, n_layers: int = 2, quantum_device: str = "default.qubit", shots=None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_qubits = nn.Linear(128, n_qubits)
        self.quantum = QuantumLayerVQE(n_qubits=n_qubits, n_layers=n_layers, num_classes=num_classes, device_name=quantum_device, shots=shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x).view(x.size(0), -1)
        angles = torch.tanh(self.to_qubits(f)) * math.pi
        return self.quantum(angles)


# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, all_y, all_p = [], [], []
    for imgs, ys in tqdm(loader, desc="Train", total=len(loader), leave=False):
        imgs = imgs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        all_y.append(ys.detach().cpu())
        all_p.append(preds.detach().cpu())
    all_y = torch.cat(all_y).numpy()
    all_p = torch.cat(all_p).numpy()
    acc = accuracy_score(all_y, all_p)
    f1m = f1_score(all_y, all_p, average="macro", zero_division=0)
    return float(np.mean(losses)), float(acc), float(f1m)


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    losses, all_y, all_p = [], [], []
    for imgs, ys in tqdm(loader, desc="Eval", total=len(loader), leave=False):
        imgs = imgs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, ys)
        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        all_y.append(ys.detach().cpu())
        all_p.append(preds.detach().cpu())
    all_y = torch.cat(all_y).numpy()
    all_p = torch.cat(all_p).numpy()
    acc = accuracy_score(all_y, all_p)
    f1m = f1_score(all_y, all_p, average="macro", zero_division=0)
    report = classification_report(all_y, all_p, target_names=class_names, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_y, all_p)
    return float(np.mean(losses)), float(acc), float(f1m), report, cm


def main():
    ap = argparse.ArgumentParser("QCNN VQE • Seeded • Standardized")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--test_dir", default=None)

    ap.add_argument("--dataset_id", default="dataset")
    ap.add_argument("--scenario", default="balanced")
    ap.add_argument("--split_id", default=None)
    ap.add_argument("--runs_root", default="./runs")
    ap.add_argument("--run_tag", default=None)

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight_decay", type=float, default=0.0005)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--label_smoothing", type=float, default=0.01)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])

    ap.add_argument("--class_weights", default="auto", choices=["none", "auto"])

    ap.add_argument("--n_qubits", type=int, default=6)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--quantum_device", default="default.qubit")
    ap.add_argument("--shots", default=None)

    args = ap.parse_args()

    set_seed_all(args.seed, deterministic_cudnn=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[WARN] CUDA non dispo -> CPU")
    else:
        device = torch.device(args.device)

    # Run dir
    model_id = "qcnn_vqe"
    run_id = args.run_tag
    paths = make_run_dir(args.runs_root, args.dataset_id, args.scenario, task="cls", model_id=model_id, run_id=run_id)

    # Sécurité : s'assurer que l'arborescence existe (même si make_run_dir le fait déjà)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Datasets (deterministic transforms; no augmentation here)
    tf = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
    train_ds = ImageFolder(args.train_dir, transform=tf)
    val_ds = ImageFolder(args.val_dir, transform=tf)
    test_ds = ImageFolder(args.test_dir, transform=tf) if args.test_dir else None

    # Remap to canonical order
    remap_imagefolder_to_canonical(train_ds)
    remap_imagefolder_to_canonical(val_ds)
    if test_ds is not None:
        remap_imagefolder_to_canonical(test_ds)

    class_names = CANONICAL_NAMES
    num_classes = len(class_names)

    # DataLoaders
    y_train = [y for _, y in train_ds.samples]
    sampler = BalancedBatchSampler(y_train, batch_size=args.batch_size, drop_last=True, resample_short=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=(device.type == "cuda"))
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=(device.type == "cuda"))

    # Class weights
    cnt = Counter(y_train)
    if args.class_weights == "auto":
        total_train = sum(cnt.values())
        weights = torch.tensor([total_train / (num_classes * max(1, cnt.get(c, 0))) for c in range(num_classes)], dtype=torch.float32, device=device)
        criterion = FocalSmoothLoss(gamma=args.gamma, eps=args.label_smoothing, weight=weights)
    else:
        weights = None
        criterion = FocalSmoothLoss(gamma=args.gamma, eps=args.label_smoothing, weight=None)

    # Model
    shots = None
    if args.shots is not None and str(args.shots).lower() != "none":
        shots = int(args.shots)
    model = HybridQCNNVQE(num_classes=num_classes, n_qubits=args.n_qubits, n_layers=args.n_layers, quantum_device=args.quantum_device, shots=shots).to(device)
    # qparams on cpu/double is handled inside forward by qml; keep as is.

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Logs
    metrics_csv = paths.logs_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "train_f1m", "val_loss", "val_acc", "val_f1m"])

    best_f1 = -1.0
    best_epoch = -1
    best_ckpt = paths.artifacts_dir / "best_qcnn_vqe.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1, va_report, va_cm = evaluate(model, val_loader, criterion, device, class_names)

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{tr_f1:.6f}", f"{va_loss:.6f}", f"{va_acc:.6f}", f"{va_f1:.6f}"])

        # dump raw val artifacts
        write_json(paths.raw_dir / f"val_report_epoch{epoch}.json", va_report)
        np.savetxt(paths.raw_dir / f"val_confusion_epoch{epoch}.csv", va_cm, fmt="%d", delimiter=",")

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_f1_macro": best_f1,
                    "class_names": class_names,
                    "args": vars(args),
                },
                best_ckpt,
            )

    # Load best and evaluate once more (val + optional test)
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    val_loss, val_acc, val_f1, val_report, val_cm = evaluate(model, val_loader, criterion, device, class_names)
    write_json(paths.raw_dir / "val_report_best.json", val_report)
    np.savetxt(paths.raw_dir / "val_confusion_best.csv", val_cm, fmt="%d", delimiter=",")

    test_metrics = {}
    if test_loader is not None:
        te_loss, te_acc, te_f1, te_report, te_cm = evaluate(model, test_loader, criterion, device, class_names)
        write_json(paths.raw_dir / "test_report.json", te_report)
        np.savetxt(paths.raw_dir / "test_confusion.csv", te_cm, fmt="%d", delimiter=",")
        test_metrics = {"test_loss": te_loss, "test_acc": te_acc, "test_f1m": te_f1}

    # metrics.json payload
    hparams = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gamma": args.gamma,
        "label_smoothing": args.label_smoothing,
        "class_weights": args.class_weights,
        "n_qubits": args.n_qubits,
        "n_layers": args.n_layers,
        "quantum_device": args.quantum_device,
        "shots": shots,
        "best_epoch": best_epoch,
        "selection_metric": "val_macro_f1",
        "canonical_class_order": CANONICAL_NAMES,
        "variant": "VQE_features(n_qubits)+energy",
    }
    metrics = {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1m": val_f1,
        **test_metrics,
    }

    # split hash (optional): we hash file paths for traceability
    split_hash = None
    try:
        from experiment_utils import sha256_paths
        split_hash = sha256_paths([p for p, _ in train_ds.samples] + [p for p, _ in val_ds.samples])
    except Exception:
        split_hash = None

    payload = canonical_metrics_schema(
        task="cls",
        model_id=model_id,
        dataset_id=args.dataset_id,
        scenario=args.scenario,
        split_id=args.split_id,
        split_hash=split_hash,
        seed=args.seed,
        hparams=hparams,
        metrics=metrics,
        paths=paths,
    )
    save_metrics_json(paths, payload)

    # Also write a small summary for convenience
    summary = {
        "best_val_f1m": best_f1,
        "best_epoch": best_epoch,
        "val_f1m_best": val_f1,
        **test_metrics,
    }
    write_json(paths.run_dir / "summary.json", summary)

    print("[DONE] run_dir:", paths.run_dir)
    print("[DONE] best_val_f1m:", best_f1)
    if test_loader is not None:
        print("[DONE] test_f1m:", test_metrics.get("test_f1m"))


if __name__ == "__main__":
    main()
