#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QCNN (CNN -> Quantum layer, variante VQE) • Reproductible + anti-biais visuel
• Seed global + cuDNN déterministe + workers=0 + AMP off (implicite)
• BalancedBatchSampler (lots équilibrés par classe)
• Focal + Label Smoothing (γ=1.5, ε=0.01) + class_weights (auto)
• AdamW (lr=0.002, wd=0.0005) aligné avec YOLO/RT-DETR
• Export métriques val (rapport par classe + matrice de conf)
• Couche quantique VQE qui encode les angles issus du CNN et renvoie [⟨Z₀⟩..⟨Z_{n-1}⟩, ⟨H⟩]
"""

import os, math, csv, json, time, random
from pathlib import Path
from collections import Counter, defaultdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm
import pennylane as qml

# ——————————————————————————
# Reproductibilité stricte
# ——————————————————————————
def set_seed_all(seed: int = 42, deterministic_cudnn: bool = True):
    print(f"[INFO] Seed={seed} | cuDNN deterministic={deterministic_cudnn}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)  # exploite quelques coeurs CPU sans casser la repro

def now():
    return time.strftime("%Y-%m-%d_%H-%M-%S")

# ——————————————————————————
# Sampler équilibré (flex)
# ——————————————————————————
class BalancedBatchSampler(Sampler[List[int]]):
    """Batches équilibrés entre classes. Tolère batch_size non multiple du nb de classes en complétant."""
    def __init__(self, labels, batch_size, drop_last=True, resample_short=False, seed=None):
        super().__init__(data_source=None)
        self.labels = list(labels)
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
            raise ValueError("BalancedBatchSampler nécessite au moins 2 classes.")
        if self.batch_size < self.num_classes:
            raise ValueError(f"batch_size={self.batch_size} < nb_classes={self.num_classes}")

        self.samples_per_class = self.batch_size // self.num_classes
        self.remainder = self.batch_size - self.samples_per_class * self.num_classes
        self._reset_epoch_state()

    def _reset_epoch_state(self):
        self.perm, self.ptr = {}, {}
        for c in self.classes:
            perm = list(self.class_to_indices[c])
            self.rng.shuffle(perm)
            self.perm[c] = perm
            self.ptr[c] = 0

    def __iter__(self):
        self._reset_epoch_state()
        while True:
            batch = []
            # K par classe
            for c in self.classes:
                need = self.samples_per_class
                got = 0
                while got < need:
                    if self.ptr[c] >= len(self.perm[c]):
                        if not self.resample_short:
                            if not self.drop_last and batch:
                                yield batch
                            return
                        self.rng.shuffle(self.perm[c]); self.ptr[c] = 0
                    batch.append(self.perm[c][self.ptr[c]])
                    self.ptr[c] += 1; got += 1

            # compléter si batch non multiple
            if self.remainder > 0:
                c_idx = 0
                for _ in range(self.remainder):
                    c = self.classes[c_idx % self.num_classes]
                    if self.ptr[c] >= len(self.perm[c]):
                        if not self.resample_short:
                            if not self.drop_last and batch:
                                yield batch
                            return
                        self.rng.shuffle(self.perm[c]); self.ptr[c] = 0
                    batch.append(self.perm[c][self.ptr[c]])
                    self.ptr[c] += 1; c_idx += 1

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

# ——————————————————————————
# Couche quantique (VQE) + modèle
# ——————————————————————————
class QuantumLayerVQE(nn.Module):
    """
    Variante VQE :
    - Encode les angles issus du CNN (data-encoding indispensable pour la discrimination)
    - Hamiltonien H = Σ_i Z_i (simple et différentiable avec adjoint)
    - Retourne un vecteur features de taille n_qubits + 1 : [⟨Z₀⟩..⟨Z_{n-1}⟩, ⟨H⟩]
    """
    def __init__(self, n_qubits: int, n_layers: int, num_classes: int,
                 device_name: str = "default.qubit", shots=None):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.num_classes = int(num_classes)

        # Device PennyLane
        self.qdev = qml.device(device_name, wires=self.n_qubits, shots=shots)

        # Paramètres variat. du circuit (double rotation par qubit et par couche)
        self.qparams = nn.Parameter(torch.zeros(self.n_layers, self.n_qubits, 2, dtype=torch.float64))

        # Hamiltonien simple : somme de PauliZ (VQE jouet mais utile comme feature)
        coeffs = [1.0] * self.n_qubits
        obs = [qml.PauliZ(i) for i in range(self.n_qubits)]
        self.hamiltonian = qml.Hamiltonian(coeffs, obs)

        @qml.qnode(self.qdev, interface="torch", diff_method="adjoint")
        def circuit(inputs, weights):
            # Préparation + encodage données
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)
                qml.RY(inputs[w], wires=w)  # encodage data indispens. pour la classification
            # L couches : anneau CNOT + rotations paramétriques
            for l in range(self.n_layers):
                for w in range(self.n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % self.n_qubits])
                for w in range(self.n_qubits):
                    qml.RZ(weights[l, w, 0], wires=w)
                    qml.RY(weights[l, w, 1], wires=w)
            # Renvoie à la fois les ⟨Z_i⟩ ET l'énergie ⟨H⟩
            z_exps = [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]
            energy = qml.expval(self.hamiltonian)
            return z_exps + [energy]

        self.circuit = circuit
        self.head = nn.Linear(self.n_qubits + 1, self.num_classes)

    def forward(self, x_angles):  # x_angles: (B, n_qubits) en radians
        outs = []
        B = x_angles.shape[0]
        for i in range(B):
            x_cpu = x_angles[i].to("cpu", dtype=torch.float64)
            out_list = self.circuit(x_cpu, self.qparams)  # longueur n_qubits+1
            out_t = torch.stack(list(out_list), dim=0) if isinstance(out_list, (list, tuple)) else out_list
            out_t = out_t.to(device=x_angles.device, dtype=x_angles.dtype)
            outs.append(out_t.view(-1))
        feats = torch.stack(outs, dim=0)  # (B, n_qubits+1)
        return self.head(feats)

class HybridQCNN(nn.Module):
    def __init__(self, num_classes: int, n_qubits: int = 6, n_layers: int = 2,
                 quantum_device="default.qubit", shots=None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_qubits = nn.Linear(128, n_qubits)
        # >>> Utilise la variante VQE <<<
        self.quantum = QuantumLayerVQE(
            n_qubits=n_qubits, n_layers=n_layers, num_classes=num_classes,
            device_name=quantum_device, shots=shots
        )

    def forward(self, x):
        f = self.features(x).view(x.size(0), -1)        # (B,128)
        angles = torch.tanh(self.to_qubits(f)) * math.pi
        return self.quantum(angles)

# ——————————————————————————
# Perte Focal + Label Smoothing
# ——————————————————————————
class FocalSmoothLoss(nn.Module):
    """Focal Cross-Entropy + Label Smoothing (γ=1.5, ε=0.01), pondérable par poids de classe."""
    def __init__(self, gamma: float = 1.5, eps: float = 0.01, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        C = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.eps / C)
            true_dist.scatter_(1, targets.unsqueeze(1), 1 - self.eps + self.eps / C)
        ce = -(true_dist * log_probs).sum(dim=1)   # (B,)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.weight is not None:
            w = self.weight[targets]
            loss = loss * w
        return loss.mean()

# ——————————————————————————
# Train / Eval
# ——————————————————————————
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, all_y, all_p = [], [], []
    for imgs, ys in tqdm(loader, desc="Train", total=len(loader), leave=False):
        imgs = imgs.to(device, non_blocking=True)
        ys   = ys.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        all_y.append(ys.detach().cpu()); all_p.append(preds.detach().cpu())
    all_y = torch.cat(all_y).numpy(); all_p = torch.cat(all_p).numpy()
    acc = accuracy_score(all_y, all_p)
    f1m = f1_score(all_y, all_p, average="macro", zero_division=0)
    return np.mean(losses), acc, f1m

@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    losses, all_y, all_p = [], [], []
    for imgs, ys in tqdm(loader, desc="Val", total=len(loader), leave=False):
        imgs = imgs.to(device, non_blocking=True)
        ys   = ys.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, ys)
        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        all_y.append(ys.detach().cpu()); all_p.append(preds.detach().cpu())
    all_y = torch.cat(all_y).numpy(); all_p = torch.cat(all_p).numpy()
    acc = accuracy_score(all_y, all_p)
    f1m = f1_score(all_y, all_p, average="macro", zero_division=0)
    report = classification_report(all_y, all_p, target_names=class_names,
                                   digits=3, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_y, all_p)
    return np.mean(losses), acc, f1m, report, cm

# ——————————————————————————
# Main
# ——————————————————————————
def main():
    import argparse, math
    parser = argparse.ArgumentParser("QCNN Balanced • Seeded & Deterministic (VQE variant)")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir",   type=str, required=True)
    parser.add_argument("--epochs",    type=int, default=100)
    parser.add_argument("--batch_size",type=int, default=12)       # multiple du nb de classes recommandé
    parser.add_argument("--img_size",  type=int, default=640)
    parser.add_argument("--lr",        type=float, default=0.002)  # aligné YOLO/RT-DETR
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--workers",   type=int, default=0)
    parser.add_argument("--device",    type=str, default="cuda", choices=["cuda","cpu","mps"])
    # Quantum
    parser.add_argument("--n_qubits",  type=int, default=6)
    parser.add_argument("--n_layers",  type=int, default=2)
    parser.add_argument("--q_shots",   type=lambda x: None if str(x).lower()=="none" else int(x), default="None")
    parser.add_argument("--q_device",  type=str, default="default.qubit")
    # Loss weighting
    parser.add_argument("--class_weights", type=str, default="auto", choices=["none","auto"])
    # I/O
    parser.add_argument("--outdir",    type=str, default="./runs_qcnn_seeded_vqe")
    args = parser.parse_args()

    # Seed & device
    set_seed_all(args.seed, deterministic_cudnn=True)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device} | seed={args.seed}")

    # Sorties
    outdir = Path(args.outdir) / f"qcnn_{now()}"
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Transfos déterministes
    tf = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
    train_ds = ImageFolder(args.train_dir, transform=tf)
    val_ds   = ImageFolder(args.val_dir,   transform=tf)
    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"[INFO] classes ({num_classes}) = {class_names}")

    # Répartition brute (log publi)
    y_train = [lbl for _, lbl in train_ds.samples]
    cnt = Counter(y_train)
    print("[INFO] Répartition train :", {class_names[k]: v for k, v in sorted(cnt.items())})

    # DataLoaders (repro)
    batch_sampler = BalancedBatchSampler(labels=y_train, batch_size=args.batch_size,
                                         drop_last=True, resample_short=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=0,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                              pin_memory=(device.type == "cuda"))

    # Modèle (variante VQE)
    model = HybridQCNN(num_classes=num_classes, n_qubits=args.n_qubits, n_layers=args.n_layers,
                       quantum_device=args.q_device, shots=args.q_shots).to(device)
    with torch.no_grad():
        # paramètres quantiques sur CPU/float64 (géré par PL au call)
        if hasattr(model, "quantum") and hasattr(model.quantum, "qparams"):
            model.quantum.qparams.data = model.quantum.qparams.data.cpu().double()

    # Poids de classe
    if args.class_weights == "auto":
        total_train = sum(cnt.values())
        weights = torch.tensor(
            [total_train / (num_classes * cnt[c]) for c in range(num_classes)],
            dtype=torch.float32, device=device
        )
        loss_fn = FocalSmoothLoss(gamma=1.5, eps=0.01, weight=weights)
        print(f"[INFO] class_weights(auto) = {weights.detach().cpu().numpy().round(4).tolist()}")
    else:
        loss_fn = FocalSmoothLoss(gamma=1.5, eps=0.01)

    # Optim (aligné YOLO/RT-DETR)
    other_params = [p for n, p in model.named_parameters() if n != "quantum.qparams"]
    optimizer = optim.AdamW([
        {"params": other_params,            "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": [model.quantum.qparams], "lr": args.lr, "weight_decay": args.weight_decay}
    ])

    # Fichier métriques
    history_csv = outdir / "metrics.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as fcsv:
        csv.writer(fcsv).writerow(["epoch","train_loss","train_acc","train_f1m","val_loss","val_acc","val_f1m"])

    # Sanity step (1 batch)
    try:
        xb, yb = next(iter(train_loader))
        xb, yb = xb.to(device), yb.to(device)
        t0 = time.time()
        logits = model(xb)
        sanity_loss = loss_fn(logits, yb)
        sanity_loss.backward()
        optimizer.zero_grad(set_to_none=True)
        print(f"[SANITY] 1 batch OK en {time.time()-t0:.2f}s | loss={sanity_loss.item():.4f}")
    except Exception as e:
        print("[SANITY][ERROR]", e)
        raise SystemExit(1)

    # Boucle d'entraînement
    best_f1m = -1.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        tr_loss, tr_acc, tr_f1m = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        va_loss, va_acc, va_f1m, report, cm = evaluate(model, val_loader, loss_fn, device, class_names)

        print(f"[Train] loss={tr_loss:.4f} | acc={tr_acc:.3f} | F1m={tr_f1m:.3f}")
        print(f"[Val  ] loss={va_loss:.4f} | acc={va_acc:.3f} | F1m={va_f1m:.3f}")

        with open(history_csv, "a", newline="", encoding="utf-8") as fcsv:
            csv.writer(fcsv).writerow([epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{tr_f1m:.6f}",
                                       f"{va_loss:.6f}", f"{va_acc:.6f}", f"{va_f1m:.6f}"])

        with open(outdir / f"val_report_epoch{epoch}.json", "w", encoding="utf-8") as fj:
            json.dump(report, fj, ensure_ascii=False, indent=2)
        np.savetxt(outdir / f"val_confusion_epoch{epoch}.csv", cm, fmt="%d", delimiter=",")

        if va_f1m > best_f1m:
            best_f1m = va_f1m
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1_macro": best_f1m,
                "class_names": class_names,
                "args": vars(args),
            }, outdir / "checkpoints" / "best_qcnn.pt")
            print(f"[✓] Nouveau best F1_macro={best_f1m:.4f} → {outdir / 'checkpoints' / 'best_qcnn.pt'}")

    # Résumé final
    summary = {
        "best_val_macroF1": round(float(best_f1m), 6),
        "epochs": args.epochs, "batch_size": args.batch_size, "img_size": args.img_size,
        "seed": args.seed, "n_qubits": args.n_qubits, "n_layers": args.n_layers,
        "q_device": args.q_device, "shots": None if args.q_shots is None else int(args.q_shots),
        "class_weights": args.class_weights,
        "quantum_variant": "VQE_features(n_qubits)+energy"
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, ensure_ascii=False, indent=2)

    print("\n=== Terminé ===")
    print(f"Meilleur F1_macro (val) : {best_f1m:.4f}")
    print(f"Logs / checkpoints : {outdir}")

if __name__ == "__main__":
    main()
