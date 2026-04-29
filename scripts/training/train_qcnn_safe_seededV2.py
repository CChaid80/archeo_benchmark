#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_qcnn_safe_seededV2.py (CORRIGÉ • paper-ready)

QCNN hybride (CNN -> PQC -> MLP) • entraînement reproductible

Objectif principal (papier) :
- Comparer (i) dataset déséquilibré "réalité terrain" vs (ii) dataset équilibré via augmentation,
  à protocole d'entraînement identique.
- Par défaut, aucune stratégie "cost-sensitive" n'est appliquée (imbalance_strategy=none).

Options secondaires (ablation / suppl.) :
- `--imbalance_strategy` peut activer :
    * weights          : pondération de la loss par fréquence de classe (auto)
    * sampler          : batches équilibrés (BalancedBatchSampler)
    * sampler_weights  : combinaison sampler + weights
- `--loss` : ce (par défaut) ou focal_smooth

Entrées attendues (ImageFolder) :
train_dir/
  sigillee/
  CO/
  CR/
val_dir/
  sigillee/
  CO/
  CR/
(optionnel) test_dir/ idem

Remarque : le circuit quantique est simulé (PennyLane default.qubit). La tête quantique
est paramétrée (qparams) et optimisée conjointement au reste du réseau.
"""

import os
import csv
import json
import time
import math
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from tqdm import tqdm
import pennylane as qml


# -----------------------------
# Reproductibilité
# -----------------------------
def set_seed_all(seed: int = 42, deterministic_cudnn: bool = True, nthreads: int = 4):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        torch.set_num_threads(int(nthreads))
    except Exception:
        pass

    if deterministic_cudnn:
        cudnn.deterministic = True
        cudnn.benchmark = False

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def now():
    return time.strftime("%Y-%m-%d_%H-%M-%S")


# -----------------------------
# Sampler équilibré
# -----------------------------
class BalancedBatchSampler(Sampler[List[int]]):
    """Batches équilibrés entre classes."""

    def __init__(self, labels, batch_size, drop_last=True, resample_short=False, seed: int = 42):
        super().__init__(data_source=None)
        self.labels = list(map(int, labels))
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.resample_short = bool(resample_short)
        self.rng = random.Random(int(seed))

        self.class_to_indices = defaultdict(list)
        for idx, y in enumerate(self.labels):
            self.class_to_indices[y].append(idx)

        self.classes = sorted(self.class_to_indices.keys())
        self.num_classes = len(self.classes)
        if self.num_classes <= 1:
            raise ValueError("BalancedBatchSampler nécessite au moins 2 classes.")
        if self.batch_size < self.num_classes:
            raise ValueError(f"batch_size={self.batch_size} < nb_classes={self.num_classes}")

        self.samples_per_class = self.batch_size // self.num_classes
        self.remainder = self.batch_size - self.samples_per_class * self.num_classes

    def __iter__(self):
        perm, ptr = {}, {}
        for c in self.classes:
            p = list(self.class_to_indices[c])
            self.rng.shuffle(p)
            perm[c] = p
            ptr[c] = 0

        while True:
            batch = []
            for c in self.classes:
                for _ in range(self.samples_per_class):
                    if ptr[c] >= len(perm[c]):
                        if not self.resample_short:
                            if not self.drop_last and batch:
                                yield batch
                            return
                        self.rng.shuffle(perm[c])
                        ptr[c] = 0
                    batch.append(perm[c][ptr[c]])
                    ptr[c] += 1

            if self.remainder > 0:
                ci = 0
                for _ in range(self.remainder):
                    c = self.classes[ci % self.num_classes]
                    if ptr[c] >= len(perm[c]):
                        if not self.resample_short:
                            if not self.drop_last and batch:
                                yield batch
                            return
                        self.rng.shuffle(perm[c])
                        ptr[c] = 0
                    batch.append(perm[c][ptr[c]])
                    ptr[c] += 1
                    ci += 1

            if len(batch) == self.batch_size:
                yield batch
            else:
                if not self.drop_last and batch:
                    yield batch
                return

    def __len__(self):
        min_class = min(len(v) for v in self.class_to_indices.values())
        k = max(1, self.samples_per_class)
        return max(1, min_class // k)


# -----------------------------
# Pertes
# -----------------------------
class LabelSmoothingCE(nn.Module):
    def __init__(self, eps: float = 0.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.eps = float(eps)
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.register_buffer("weight", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=1)
        loss = (1.0 - self.eps) * nll + self.eps * smooth
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


class FocalSmoothLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, eps: float = 0.0, weight: Optional[torch.Tensor] = None):
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


# -----------------------------
# Quantum layer (PennyLane)
# -----------------------------
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=6, n_layers=4, dev_name="default.qubit"):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)

        self.dev = qml.device(dev_name, wires=self.n_qubits)

        # Paramètres quantiques entraînables (CPU, float64)
        init_scale = 0.01
        qparams = torch.randn(self.n_layers, self.n_qubits, 3, dtype=torch.float64) * init_scale
        self.qparams = nn.Parameter(qparams)

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(inputs, weights):
            # inputs: (n_qubits,) float64 on CPU
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, x_angles: torch.Tensor) -> torch.Tensor:
        """
        x_angles: (B, n_qubits) sur device (cuda ou cpu).
        Circuit exécuté sur CPU (PennyLane default.qubit), sample-by-sample.
        """
        B = x_angles.shape[0]
        outs = []
        for i in range(B):
            x_cpu = x_angles[i].to("cpu", dtype=torch.float64)
            out = self.circuit(x_cpu, self.qparams)  # list/tuple -> stack
            if isinstance(out, (list, tuple)):
                out = torch.stack(list(out))
            outs.append(out.to(device=x_angles.device, dtype=x_angles.dtype))
        return torch.stack(outs, dim=0)  # (B, n_qubits)


# -----------------------------
# Modèle hybride
# -----------------------------
class HybridQCNN(nn.Module):
    def __init__(self, n_qubits=6, n_layers=4, n_classes=3):
        super().__init__()
        self.n_qubits = int(n_qubits)

        # CNN compact (RGB)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        # inférer dim
        dummy = torch.zeros(1, 3, 64, 64)
        feat_dim = self.cnn(dummy).shape[1]

        self.fc_to_qubits = nn.Linear(feat_dim, self.n_qubits)

        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
        # Head classique
        self.classifier = nn.Sequential(
            nn.Linear(self.n_qubits, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        feats = self.cnn(x)
        angles = torch.tanh(self.fc_to_qubits(feats)) * (math.pi / 2.0)
        q_out = self.quantum(angles)
        logits = self.classifier(q_out)
        return logits


# -----------------------------
# Utils dataset : mapping stable
# -----------------------------
def remap_imagefolder_classes(ds: ImageFolder, desired_order: List[str]) -> None:
    for c in desired_order:
        if c not in ds.class_to_idx:
            raise ValueError(f"Classe manquante dans le dataset : {c}")

    new_class_to_idx = {cls_name: i for i, cls_name in enumerate(desired_order)}
    for i, (path, label) in enumerate(ds.samples):
        old_class_name = ds.classes[label]
        new_label = new_class_to_idx[old_class_name]
        ds.targets[i] = new_label
        ds.samples[i] = (path, new_label)

    ds.class_to_idx = new_class_to_idx
    ds.classes = desired_order


# -----------------------------
# Train / Eval
# -----------------------------
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

        losses.append(float(loss.item()))
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
    for imgs, ys in tqdm(loader, desc="Val", total=len(loader), leave=False):
        imgs = imgs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, ys)
        losses.append(float(loss.item()))

        preds = logits.argmax(dim=1)
        all_y.append(ys.detach().cpu())
        all_p.append(preds.detach().cpu())

    all_y = torch.cat(all_y).numpy()
    all_p = torch.cat(all_p).numpy()
    acc = accuracy_score(all_y, all_p)
    f1m = f1_score(all_y, all_p, average="macro", zero_division=0)
    report = classification_report(
        all_y,
        all_p,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_y, all_p)
    return float(np.mean(losses)), float(acc), float(f1m), report, cm


def make_loader(
    ds: ImageFolder,
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
    use_balanced_sampler: bool,
) -> DataLoader:
    if use_balanced_sampler:
        labels = [lbl for _, lbl in ds.samples]
        sampler = BalancedBatchSampler(labels=labels, batch_size=batch_size, drop_last=True, seed=seed)
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        g = torch.Generator()
        g.manual_seed(seed)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=(device.type == "cuda"),
            generator=g,
        )
    return loader


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("QCNN hybride • Seeded & Deterministic (paper-ready)")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--img_size", type=int, default=64)

    parser.add_argument("--n_qubits", type=int, default=6)
    parser.add_argument("--n_q_layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--threads", type=int, default=4)

    parser.add_argument(
        "--imbalance_strategy",
        type=str,
        default="none",
        choices=["none", "weights", "sampler", "sampler_weights"],
    )
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal_smooth"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--focal_gamma", type=float, default=1.5)

    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="./runs_qcnn_seededV2")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    # Seed & device
    set_seed_all(args.seed, deterministic_cudnn=True, nthreads=args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[WARN] CUDA non dispo, utilisation du CPU.")
    else:
        device = torch.device(args.device)
    print(f"[INFO] device={device} | seed={args.seed}")

    # Sorties
    run_name = args.run_name or f"qcnnV2_{now()}"
    outdir = Path(args.outdir) / run_name
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Transforms (déterministes)
    tf = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )

    train_ds = ImageFolder(args.train_dir, transform=tf)
    val_ds = ImageFolder(args.val_dir, transform=tf)

    desired_order = ["sigillee", "CO", "CR"]
    remap_imagefolder_classes(train_ds, desired_order)
    remap_imagefolder_classes(val_ds, desired_order)

    class_names = train_ds.classes
    n_classes = len(class_names)
    print(f"[INFO] classes ({n_classes}) = {class_names}")
    print("[INFO] class_to_idx =", train_ds.class_to_idx)

    # Distribution train
    y_train = [lbl for _, lbl in train_ds.samples]
    cnt = Counter(y_train)
    print("[INFO] Répartition train :", {class_names[k]: v for k, v in sorted(cnt.items())})

    use_balanced_sampler = args.imbalance_strategy in ("sampler", "sampler_weights")
    use_class_weights = args.imbalance_strategy in ("weights", "sampler_weights")

    # Loaders
    train_loader = make_loader(
        ds=train_ds,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        seed=args.seed,
        use_balanced_sampler=use_balanced_sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = None
    if args.test_dir:
        test_ds = ImageFolder(args.test_dir, transform=tf)
        remap_imagefolder_classes(test_ds, desired_order)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )
        print("[INFO] Test set activé :", args.test_dir, f"(N={len(test_ds)})")

    # Class weights
    weights = None
    if use_class_weights:
        total_train = sum(cnt.values())
        w = [total_train / (n_classes * cnt[c]) for c in range(n_classes)]
        weights = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"[INFO] class_weights(auto) = {weights.detach().cpu().numpy().round(4).tolist()}")

    # Loss
    if args.loss == "ce":
        criterion = LabelSmoothingCE(eps=args.label_smoothing, weight=weights)
    else:
        criterion = FocalSmoothLoss(gamma=args.focal_gamma, eps=args.label_smoothing, weight=weights)

    # Modèle
    model = HybridQCNN(n_qubits=args.n_qubits, n_layers=args.n_q_layers, n_classes=n_classes).to(device)

    # NOTE : garder qparams sur CPU en float64 (PennyLane)
    model.quantum.qparams.data = model.quantum.qparams.data.cpu().double()

    # Optimiseur
    # Paramètres quantiques (CPU float64) + le reste (device float32)
    qparams = [model.quantum.qparams]
    other_params = [p for n, p in model.named_parameters() if n != "quantum.qparams"]
    optimizer = optim.AdamW(
        [{"params": other_params}, {"params": qparams, "lr": args.lr}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Logs
    history_csv = outdir / "metrics.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as fcsv:
        csv.writer(fcsv).writerow(
            ["epoch", "train_loss", "train_acc", "train_f1m", "val_loss", "val_acc", "val_f1m"]
        )

    # Sanity
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)
    logits = model(xb)
    sanity_loss = criterion(logits, yb)
    sanity_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    print(f"[SANITY] 1 batch OK | loss={sanity_loss.item():.4f}")

    best_f1m = -1.0
    best_epoch = -1
    es_patience = max(0, int(args.early_stop_patience))
    es_bad = 0
    ckpt_path = outdir / "checkpoints" / "best_qcnnV2.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        tr_loss, tr_acc, tr_f1m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1m, report, cm = evaluate(model, val_loader, criterion, device, class_names)

        print(f"[Train] loss={tr_loss:.4f} | acc={tr_acc:.4f} | F1m={tr_f1m:.4f}")
        print(f"[Val  ] loss={va_loss:.4f} | acc={va_acc:.4f} | F1m={va_f1m:.4f}")

        with open(history_csv, "a", newline="", encoding="utf-8") as fcsv:
            csv.writer(fcsv).writerow(
                [epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{tr_f1m:.6f}", f"{va_loss:.6f}", f"{va_acc:.6f}", f"{va_f1m:.6f}"]
            )

        with open(outdir / f"val_report_epoch{epoch}.json", "w", encoding="utf-8") as fj:
            json.dump(report, fj, ensure_ascii=False, indent=2)
        np.savetxt(outdir / f"val_confusion_epoch{epoch}.csv", cm, fmt="%d", delimiter=",")

        if va_f1m > best_f1m:
            best_f1m = va_f1m
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_f1_macro": float(best_f1m),
                    "class_names": class_names,
                    "class_to_idx": {cls: i for i, cls in enumerate(class_names)},
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[✓] Nouveau best F1_macro={best_f1m:.4f} → {ckpt_path}")
            es_bad = 0
        else:
            es_bad += 1

        if es_patience > 0 and es_bad >= es_patience:
            print(f"[EARLY STOP] patience={es_patience} atteinte. Stop à epoch={epoch}.")
            break

    # Recharger best
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    test_results = None
    if test_loader is not None:
        te_loss, te_acc, te_f1m, te_report, te_cm = evaluate(model, test_loader, criterion, device, class_names)
        test_results = {"test_loss": te_loss, "test_acc": te_acc, "test_f1m": te_f1m}
        with open(outdir / "test_report.json", "w", encoding="utf-8") as fj:
            json.dump(te_report, fj, ensure_ascii=False, indent=2)
        np.savetxt(outdir / "test_confusion.csv", te_cm, fmt="%d", delimiter=",")
        print(f"\n[Test] loss={te_loss:.4f} | acc={te_acc:.4f} | F1m={te_f1m:.4f}")

    summary = {
        "best_val_macroF1": round(float(best_f1m), 6),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epoch),
        "batch_size": int(args.batch_size),
        "img_size": int(args.img_size),
        "seed": int(args.seed),
        "workers": int(args.workers),
        "threads": int(args.threads),
        "imbalance_strategy": args.imbalance_strategy,
        "loss": args.loss,
        "label_smoothing": float(args.label_smoothing),
        "focal_gamma": float(args.focal_gamma),
        "class_names": class_names,
        "class_to_idx": {cls: i for i, cls in enumerate(class_names)},
        "model": "QCNN-V2 (CNN->PQC->MLP)",
        "n_qubits": int(args.n_qubits),
        "n_q_layers": int(args.n_q_layers),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "device": str(device),
    }
    if test_results is not None:
        summary.update(test_results)

    with open(outdir / "summary.json", "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, ensure_ascii=False, indent=2)

    print("\n=== Terminé ===")
    print(f"Meilleur F1_macro (val) : {best_f1m:.4f} (epoch {best_epoch})")
    print(f"Logs / checkpoints : {outdir}")


if __name__ == "__main__":
    main()
