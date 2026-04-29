#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_mobilenet_safe_seededV3.py (CORRIGÉ • paper-ready)

MobileNetV3-Small • Entraînement reproductible • comparatif balanced vs imbalanced

Objectif principal (papier) :
- Comparer (i) dataset déséquilibré "réalité terrain" vs (ii) dataset équilibré via augmentation,
  à protocole d'entraînement identique.
- Par défaut, aucune stratégie "cost-sensitive" n'est appliquée (imbalance_strategy=none).

Options secondaires (ablation / suppl.) :
- `--imbalance_strategy` peut activer :
    * weights          : pondération de la loss par fréquence de classe (auto)
    * sampler          : batches équilibrés (BalancedBatchSampler)
    * sampler_weights  : combinaison sampler + weights

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

Sorties :
- metrics.csv (train/val loss, acc, F1 macro)
- val_report_epoch{epoch}.json, val_confusion_epoch{epoch}.csv
- summary.json + checkpoint best_mobilenet.pt
"""

import os
import csv
import json
import time
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
from torchvision import transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from tqdm import tqdm


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

    # TF32 peut introduire une petite variabilité numérique
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass

    # Non garanti compatible avec toutes les ops, donc on protège.
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
    """Batches équilibrés entre classes. Tolère batch_size non multiple du nb de classes en complétant."""

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

            # K par classe
            for c in self.classes:
                need = self.samples_per_class
                got = 0
                while got < need:
                    if ptr[c] >= len(perm[c]):
                        if not self.resample_short:
                            if not self.drop_last and batch:
                                yield batch
                            return
                        self.rng.shuffle(perm[c])
                        ptr[c] = 0
                    batch.append(perm[c][ptr[c]])
                    ptr[c] += 1
                    got += 1

            # compléter si batch non multiple
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
    """Cross-entropy avec label smoothing (ε). Supporte weights par classe."""

    def __init__(self, eps: float = 0.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.eps = float(eps)
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.register_buffer("weight", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B,C), targets: (B,)
        log_probs = torch.log_softmax(logits, dim=1)  # (B,C)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        smooth = -log_probs.mean(dim=1)  # (B,)
        loss = (1.0 - self.eps) * nll + self.eps * smooth

        if self.weight is not None:
            loss = loss * self.weight[targets]

        return loss.mean()


class FocalSmoothLoss(nn.Module):
    """Focal CE + Label smoothing. Optionnellement pondérée par poids de classe."""

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

        ce = -(true_dist * log_probs).sum(dim=1)  # (B,)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce

        if self.weight is not None:
            loss = loss * self.weight[targets]

        return loss.mean()


# -----------------------------
# Modèle
# -----------------------------
def build_mobilenet_v3_small(num_classes: int, pretrained: bool = False) -> nn.Module:
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float, float]:
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
def evaluate(model, loader, criterion, device, class_names) -> Tuple[float, float, float, dict, np.ndarray]:
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


def remap_imagefolder_classes(ds: ImageFolder, desired_order: List[str]) -> None:
    """Force un mapping stable des labels : desired_order[i] -> i, et remap samples/targets."""
    for c in desired_order:
        if c not in ds.class_to_idx:
            raise ValueError(f"Classe manquante dans le dataset : {c}")

    new_class_to_idx = {cls_name: i for i, cls_name in enumerate(desired_order)}

    # Remap samples + targets
    for i, (path, label) in enumerate(ds.samples):
        old_class_name = ds.classes[label]
        new_label = new_class_to_idx[old_class_name]
        ds.targets[i] = new_label
        ds.samples[i] = (path, new_label)

    ds.class_to_idx = new_class_to_idx
    ds.classes = desired_order


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
    parser = argparse.ArgumentParser("MobileNetV3-Small • Seeded & Deterministic (paper-ready)")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--img_size", type=int, default=640)

    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--threads", type=int, default=4)

    parser.add_argument("--pretrained", action="store_true", help="Poids ImageNet pour MobileNet.")

    parser.add_argument(
        "--imbalance_strategy",
        type=str,
        default="none",
        choices=["none", "weights", "sampler", "sampler_weights"],
        help="Stratégie anti-déséquilibre (par défaut: none pour protocole 'augmentation only').",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "focal_smooth"],
        help="Type de loss. Par défaut: ce.",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="ε label smoothing (CE).")
    parser.add_argument("--focal_gamma", type=float, default=1.5, help="γ focal (si focal_smooth).")

    parser.add_argument("--early_stop_patience", type=int, default=0, help="0 = désactivé.")
    parser.add_argument("--outdir", type=str, default="./runs_mobilenet_seeded")
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
    run_name = args.run_name or f"mobilenet_{now()}"
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
    num_classes = len(class_names)
    print(f"[INFO] classes ({num_classes}) = {class_names}")
    print("[INFO] class_to_idx =", train_ds.class_to_idx)

    # Répartition train
    y_train = [lbl for _, lbl in train_ds.samples]
    cnt = Counter(y_train)
    print("[INFO] Répartition train :", {class_names[k]: v for k, v in sorted(cnt.items())})

    # Imbalance handling flags
    use_balanced_sampler = args.imbalance_strategy in ("sampler", "sampler_weights")
    use_class_weights = args.imbalance_strategy in ("weights", "sampler_weights")

    # DataLoaders
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

    # Optional test loader
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

    # Poids de classes (auto) si demandé
    weights = None
    if use_class_weights:
        total_train = sum(cnt.values())
        w = [total_train / (num_classes * cnt[c]) for c in range(num_classes)]
        weights = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"[INFO] class_weights(auto) = {weights.detach().cpu().numpy().round(4).tolist()}")

    # Loss
    if args.loss == "ce":
        loss_fn = LabelSmoothingCE(eps=args.label_smoothing, weight=weights)
    else:
        loss_fn = FocalSmoothLoss(gamma=args.focal_gamma, eps=args.label_smoothing, weight=weights)

    # Modèle
    model = build_mobilenet_v3_small(num_classes=num_classes, pretrained=args.pretrained).to(device)

    # Optimiseur
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Logs
    history_csv = outdir / "metrics.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as fcsv:
        csv.writer(fcsv).writerow(
            ["epoch", "train_loss", "train_acc", "train_f1m", "val_loss", "val_acc", "val_f1m"]
        )

    # Sanity step
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)
    logits = model(xb)
    sanity_loss = loss_fn(logits, yb)
    sanity_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    print(f"[SANITY] 1 batch OK | loss={sanity_loss.item():.4f}")

    # Training loop
    best_f1m = -1.0
    best_epoch = -1
    es_patience = max(0, int(args.early_stop_patience))
    es_bad = 0

    ckpt_path = outdir / "checkpoints" / "best_mobilenet.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        tr_loss, tr_acc, tr_f1m = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        va_loss, va_acc, va_f1m, report, cm = evaluate(model, val_loader, loss_fn, device, class_names)

        print(f"[Train] loss={tr_loss:.4f} | acc={tr_acc:.4f} | F1m={tr_f1m:.4f}")
        print(f"[Val  ] loss={va_loss:.4f} | acc={va_acc:.4f} | F1m={va_f1m:.4f}")

        # Append CSV
        with open(history_csv, "a", newline="", encoding="utf-8") as fcsv:
            csv.writer(fcsv).writerow(
                [epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{tr_f1m:.6f}", f"{va_loss:.6f}", f"{va_acc:.6f}", f"{va_f1m:.6f}"]
            )

        # Save per-epoch reports
        with open(outdir / f"val_report_epoch{epoch}.json", "w", encoding="utf-8") as fj:
            json.dump(report, fj, ensure_ascii=False, indent=2)
        np.savetxt(outdir / f"val_confusion_epoch{epoch}.csv", cm, fmt="%d", delimiter=",")

        # Best checkpoint (macro-F1)
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

    # Recharger meilleur checkpoint pour test éventuel
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    test_results = None
    if test_loader is not None:
        te_loss, te_acc, te_f1m, te_report, te_cm = evaluate(model, test_loader, loss_fn, device, class_names)
        test_results = {
            "test_loss": te_loss,
            "test_acc": te_acc,
            "test_f1m": te_f1m,
        }
        with open(outdir / "test_report.json", "w", encoding="utf-8") as fj:
            json.dump(te_report, fj, ensure_ascii=False, indent=2)
        np.savetxt(outdir / "test_confusion.csv", te_cm, fmt="%d", delimiter=",")
        print(f"\n[Test] loss={te_loss:.4f} | acc={te_acc:.4f} | F1m={te_f1m:.4f}")

    # Résumé final
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
        "model": "MobileNetV3-Small",
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "pretrained": bool(args.pretrained),
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
