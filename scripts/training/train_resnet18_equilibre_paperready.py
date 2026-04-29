#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_resnet18_equilibre.py

ResNet-18 • entraînement reproductible + sorties standardisées (paper-ready)

Objectif :
  - partir de tes scripts ResNet-18 (équilibré/déséquilibré)
  - supprimer les chemins hardcodés Windows et les augmentations aléatoires
  - aligner les artefacts de sortie sur les scripts standardisés :
      run_dir/
        artifacts/ (checkpoints)
        logs/      (metrics.csv, DONE.txt)
        raw/       (reports JSON, confusions CSV)
        metrics.json  (pivot pour agrégation)

Mapping de classes forcé (robuste aux accents/variantes) :
  0 = sigillee
  1 = CO
  2 = CR

Exemple :
  python train_resnet18_equilibre.py \
    --train_dir ".../crops_balanced/train" \
    --val_dir   ".../crops_balanced/val" \
    --runs_root "./runs_paper" \
    --dataset_id "arkeocera" --scenario "balanced" --split_id "splitA" \
    --epochs 80 --batch_size 12 --img_size 640 --seed 42 --pretrained
"""

from __future__ import annotations

import argparse
import csv
import random
import time
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm

from experiment_utils import (
    canonical_metrics_schema,
    make_run_dir,
    save_metrics_json,
    sha256_paths,
    write_json,
    write_text,
)

# -----------------------------
# Reproductibilité
# -----------------------------
def set_seed_all(seed: int = 42, deterministic_cudnn: bool = True, nthreads: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        cudnn.deterministic = True
        cudnn.benchmark = False
    try:
        torch.set_num_threads(nthreads)
    except Exception:
        pass


def now() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


# -----------------------------
# Normalisation des noms de classes
# -----------------------------
def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch.isalnum())
    return s


CANONICAL_ORDER: List[str] = ["sigillee", "CO", "CR"]
CANONICAL_TO_IDX: Dict[str, int] = {"sigillee": 0, "CO": 1, "CR": 2}

ALIAS_TO_CANONICAL: Dict[str, str] = {
    # sigillée
    "sigillee": "sigillee",
    "sigillees": "sigillee",
    "sig": "sigillee",
    "sigille": "sigillee",
    # CO
    "co": "CO",
    "communeoxydante": "CO",
    "oxydante": "CO",
    "communeoxydant": "CO",
    # CR
    "cr": "CR",
    "communereductrice": "CR",
    "reductrice": "CR",
    "communereducteur": "CR",
}


def remap_imagefolder_to_canonical(ds: ImageFolder) -> None:
    old_classes = list(ds.classes)

    seen = set()
    for cname in old_classes:
        key = _norm_name(cname)
        if key not in ALIAS_TO_CANONICAL:
            raise ValueError(
                f"Classe inconnue '{cname}'. "
                f"Attendu une variante de {CANONICAL_ORDER} (aliases: {sorted(ALIAS_TO_CANONICAL.keys())})."
            )
        seen.add(ALIAS_TO_CANONICAL[key])

    missing = [c for c in CANONICAL_ORDER if c not in seen]
    if missing:
        raise ValueError(f"Classes manquantes dans le dataset: {missing} (présentes: {sorted(seen)})")

    for i, (path, old_label) in enumerate(ds.samples):
        old_name = old_classes[old_label]
        canon_name = ALIAS_TO_CANONICAL[_norm_name(old_name)]
        new_label = CANONICAL_TO_IDX[canon_name]
        ds.samples[i] = (path, new_label)

    ds.targets = [lbl for _, lbl in ds.samples]
    ds.classes = list(CANONICAL_ORDER)
    ds.class_to_idx = dict(CANONICAL_TO_IDX)


def compute_split_hash(ds: ImageFolder) -> str:
    return sha256_paths([p for p, _ in ds.samples])


# -----------------------------
# Sampler équilibré (batches)
# -----------------------------
class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, labels: List[int], batch_size: int, seed: int = 42, drop_last: bool = True):
        super().__init__(data_source=None)
        self.labels = list(labels)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.rng = random.Random(seed)

        self.class_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, y in enumerate(self.labels):
            self.class_to_indices[int(y)].append(idx)

        self.classes = sorted(self.class_to_indices.keys())
        self.num_classes = len(self.classes)
        if self.num_classes <= 1:
            raise ValueError("BalancedBatchSampler nécessite au moins 2 classes.")
        if self.batch_size < self.num_classes:
            raise ValueError(f"batch_size={self.batch_size} < nb_classes={self.num_classes}")

        self.samples_per_class = self.batch_size // self.num_classes
        self.remainder = self.batch_size - self.samples_per_class * self.num_classes

    def __iter__(self):
        per_class = {}
        ptr = {}
        for c in self.classes:
            perm = list(self.class_to_indices[c])
            self.rng.shuffle(perm)
            per_class[c] = perm
            ptr[c] = 0

        while True:
            batch: List[int] = []
            for c in self.classes:
                need = self.samples_per_class
                got = 0
                while got < need:
                    if ptr[c] >= len(per_class[c]):
                        if not self.drop_last and batch:
                            yield batch
                        return
                    batch.append(per_class[c][ptr[c]])
                    ptr[c] += 1
                    got += 1

            if self.remainder > 0:
                c_idx = 0
                for _ in range(self.remainder):
                    c = self.classes[c_idx % self.num_classes]
                    if ptr[c] >= len(per_class[c]):
                        if not self.drop_last and batch:
                            yield batch
                        return
                    batch.append(per_class[c][ptr[c]])
                    ptr[c] += 1
                    c_idx += 1

            if len(batch) == self.batch_size:
                yield batch
            else:
                if not self.drop_last and batch:
                    yield batch
                return

    def __len__(self) -> int:
        min_class = min(len(v) for v in self.class_to_indices.values())
        k = max(1, self.samples_per_class)
        return max(1, min_class // k)


# -----------------------------
# Loss : Focal + Label Smoothing
# -----------------------------
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


# -----------------------------
# Modèle : ResNet-18
# -----------------------------
def build_resnet18(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float, float]:
    model.train()
    losses: List[float] = []
    all_y, all_p = [], []

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

    y = torch.cat(all_y).numpy()
    p = torch.cat(all_p).numpy()
    acc = accuracy_score(y, p)
    f1m = f1_score(y, p, average="macro", zero_division=0)
    return float(np.mean(losses)), float(acc), float(f1m)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, class_names: List[str]) -> Tuple[float, float, float, Dict, np.ndarray]:
    model.eval()
    losses: List[float] = []
    all_y, all_p = [], []

    for imgs, ys in tqdm(loader, desc="Eval", total=len(loader), leave=False):
        imgs = imgs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, ys)
        losses.append(float(loss.item()))

        preds = logits.argmax(dim=1)
        all_y.append(ys.detach().cpu())
        all_p.append(preds.detach().cpu())

    y = torch.cat(all_y).numpy()
    p = torch.cat(all_p).numpy()
    acc = accuracy_score(y, p)
    f1m = f1_score(y, p, average="macro", zero_division=0)
    report = classification_report(y, p, target_names=class_names, digits=3, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, p)
    return float(np.mean(losses)), float(acc), float(f1m), report, cm


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("ResNet-18 • Seeded & Paper-ready")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--test_dir", default=None)

    ap.add_argument("--runs_root", default="./runs")
    ap.add_argument("--dataset_id", default="dataset")
    ap.add_argument("--scenario", default="balanced")
    ap.add_argument("--split_id", default=None)
    ap.add_argument("--run_tag", default=None, help="Optionnel: ajout dans le run_id")

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight_decay", type=float, default=0.0005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])

    ap.add_argument("--class_weights", type=str, default="auto", choices=["none", "auto"])
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--label_smoothing", type=float, default=0.01)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Seed & device
    set_seed_all(args.seed, deterministic_cudnn=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[WARN] CUDA non dispo, utilisation du CPU.")
    else:
        device = torch.device(args.device)
    print(f"[INFO] device={device} | seed={args.seed}")

    # Arborescence run
    model_id = "resnet18"
    run_id = now()
    if args.run_tag:
        run_id = f"{run_id}_{args.run_tag}"

    paths = make_run_dir(
        runs_root=args.runs_root,
        dataset_id=args.dataset_id,
        scenario=args.scenario,
        task="cls",
        model_id=model_id,
        run_id=run_id,
    )

    # Transforms (sans augmentation)
    tf = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])

    train_ds = ImageFolder(args.train_dir, transform=tf)
    val_ds = ImageFolder(args.val_dir, transform=tf)
    test_ds = ImageFolder(args.test_dir, transform=tf) if args.test_dir else None

    remap_imagefolder_to_canonical(train_ds)
    remap_imagefolder_to_canonical(val_ds)
    if test_ds is not None:
        remap_imagefolder_to_canonical(test_ds)

    class_names = list(train_ds.classes)
    num_classes = len(class_names)

    # Hash split
    split_hash = compute_split_hash(train_ds) + ":" + compute_split_hash(val_ds)
    if test_ds is not None:
        split_hash += ":" + compute_split_hash(test_ds)

    # Distrib train (log)
    y_train = [lbl for _, lbl in train_ds.samples]
    cnt = Counter(y_train)
    dist = {class_names[k]: int(v) for k, v in sorted(cnt.items())}
    print("[INFO] Répartition train:", dist)

    write_json(paths.run_dir / "dist_train.json", dist)
    write_json(paths.run_dir / "classes.json", {"class_names": class_names, "class_to_idx": train_ds.class_to_idx})

    # DataLoaders
    batch_sampler = BalancedBatchSampler(labels=y_train, batch_size=args.batch_size, seed=args.seed, drop_last=True)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )

    # Class weights
    if args.class_weights == "auto":
        total_train = sum(cnt.values())
        weights = torch.tensor(
            [total_train / (num_classes * cnt[c]) for c in range(num_classes)],
            dtype=torch.float32,
            device=device,
        )
        loss_fn = FocalSmoothLoss(gamma=args.gamma, eps=args.label_smoothing, weight=weights)
        print(f"[INFO] class_weights(auto)={weights.detach().cpu().numpy().round(4).tolist()}")
    else:
        loss_fn = FocalSmoothLoss(gamma=args.gamma, eps=args.label_smoothing, weight=None)
        print("[INFO] class_weights=none")

    # Modèle
    model = build_resnet18(num_classes=num_classes, pretrained=args.pretrained).to(device)

    # Optimiseur
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # CSV métriques
    history_csv = paths.logs_dir / "metrics.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "train_f1m", "val_loss", "val_acc", "val_f1m"])

    # Train loop
    best_f1m = -1.0
    best_epoch = -1
    best_ckpt = paths.artifacts_dir / "best_resnet18.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        tr_loss, tr_acc, tr_f1m = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        va_loss, va_acc, va_f1m, va_report, va_cm = evaluate(model, val_loader, loss_fn, device, class_names)

        with open(history_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{tr_f1m:.6f}", f"{va_loss:.6f}", f"{va_acc:.6f}", f"{va_f1m:.6f}"]
            )

        write_json(paths.raw_dir / f"val_report_epoch{epoch}.json", va_report)
        np.savetxt(paths.raw_dir / f"val_confusion_epoch{epoch}.csv", va_cm, fmt="%d", delimiter=",")

        print(f"[Train] loss={tr_loss:.4f} | acc={tr_acc:.3f} | F1m={tr_f1m:.3f}")
        print(f"[Val  ] loss={va_loss:.4f} | acc={va_acc:.3f} | F1m={va_f1m:.3f}")

        if va_f1m > best_f1m:
            best_f1m = float(va_f1m)
            best_epoch = int(epoch)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_macro_f1": best_f1m,
                    "class_names": class_names,
                    "args": vars(args),
                },
                best_ckpt,
            )
            print(f"[✓] Nouveau best macro-F1={best_f1m:.4f} @epoch={best_epoch} -> {best_ckpt.name}")

    # Reload best + final evaluation
    print("\n[INFO] Reload meilleur checkpoint pour évaluation finale…")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    if test_loader is not None:
        split_name = "test"
        te_loss, te_acc, te_f1m, te_report, te_cm = evaluate(model, test_loader, loss_fn, device, class_names)
        write_json(paths.raw_dir / "test_report.json", te_report)
        np.savetxt(paths.raw_dir / "test_confusion.csv", te_cm, fmt="%d", delimiter=",")
        final_metrics = {"loss": te_loss, "acc": te_acc, "macro_f1": te_f1m}
    else:
        split_name = "val"
        va_loss, va_acc, va_f1m, va_report, va_cm = evaluate(model, val_loader, loss_fn, device, class_names)
        write_json(paths.raw_dir / "val_report_best.json", va_report)
        np.savetxt(paths.raw_dir / "val_confusion_best.csv", va_cm, fmt="%d", delimiter=",")
        final_metrics = {"loss": va_loss, "acc": va_acc, "macro_f1": va_f1m}

    hparams = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gamma": args.gamma,
        "label_smoothing": args.label_smoothing,
        "class_weights": args.class_weights,
        "pretrained": bool(args.pretrained),
        "workers": args.workers,
        "device": str(device),
        "selection": {"criterion": "best_val_macro_f1", "best_epoch": best_epoch, "best_val_macro_f1": float(best_f1m)},
        "final_eval_split": split_name,
    }
    metrics = {
        "final": final_metrics,
        "class_names": class_names,
        "best_val_macro_f1": float(best_f1m),
    }
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
    mj = save_metrics_json(paths, payload)
    write_text(paths.logs_dir / "DONE.txt", f"done\nmetrics_json={mj}\n")
    print(f"[OK] metrics.json: {mj}")


if __name__ == "__main__":
    main()
