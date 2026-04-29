# -*- coding: utf-8 -*-
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import classification_report

import pennylane as qml

CANON = ["sigillee", "CO", "CR"]

# =========================
# QUANTUM (IDENTIQUE TRAIN)
# =========================
class QuantumLayerVQE(nn.Module):
    def __init__(self, n_qubits=6, n_layers=2):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.qdev = qml.device("default.qubit", wires=n_qubits)

        self.qparams = nn.Parameter(torch.zeros(n_layers, n_qubits, 2, dtype=torch.float64))

        coeffs = [1.0] * n_qubits
        obs = [qml.PauliZ(i) for i in range(n_qubits)]
        self.hamiltonian = qml.Hamiltonian(coeffs, obs)

        self.head = nn.Linear(n_qubits + 1, 3)

        @qml.qnode(self.qdev, interface="torch", diff_method="adjoint")
        def circuit(inputs, weights):
            for w in range(n_qubits):
                qml.Hadamard(wires=w)
                qml.RY(inputs[w], wires=w)

            for l in range(n_layers):
                for w in range(n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % n_qubits])
                for w in range(n_qubits):
                    qml.RZ(weights[l, w, 0], wires=w)
                    qml.RY(weights[l, w, 1], wires=w)

            z_exps = [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
            energy = qml.expval(self.hamiltonian)

            return z_exps + [energy]

        self.circuit = circuit

    def forward(self, x):
        outs = []
        for i in range(x.shape[0]):
            xi = x[i].to("cpu", dtype=torch.float64)
            o = self.circuit(xi, self.qparams)
            o = torch.stack(list(o)).to(x.device, dtype=x.dtype)
            outs.append(o)

        feats = torch.stack(outs)
        return self.head(feats)

# =========================
# MODELE VQE
# =========================
class HybridQCNNVQE(nn.Module):
    def __init__(self, n_qubits=6, n_layers=2, n_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.to_qubits = nn.Linear(128, n_qubits)
        self.quantum = QuantumLayerVQE(n_qubits, n_layers)

    def forward(self, x):
        f = self.features(x).view(x.size(0), -1)
        angles = torch.tanh(self.to_qubits(f)) * math.pi
        return self.quantum(angles)

# =========================
# DATASET
# =========================
def remap(ds):
    mapping = {c: i for i, c in enumerate(CANON)}
    for i, (p, y) in enumerate(ds.samples):
        name = ds.classes[y]
        ds.samples[i] = (p, mapping[name])
        ds.targets[i] = mapping[name]
    ds.classes = CANON
    ds.class_to_idx = mapping

# =========================
# EVAL
# =========================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    yt, yp = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()

        yt.extend(y.numpy())
        yp.extend(preds)

    return classification_report(
        yt, yp,
        target_names=CANON,
        output_dict=True,
        digits=6,
        zero_division=0
    )

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    ds = ImageFolder(args.val_dir, transform=tf)
    remap(ds)

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    model = HybridQCNNVQE(n_layers=args.n_layers)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)

    rep = evaluate(model, loader, device)

    Path(args.out_json).write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("\n=== RESULTATS ===")
    print("F1(sigillee) =", rep["sigillee"]["f1-score"])
    print("F1(CO)       =", rep["CO"]["f1-score"])
    print("F1(CR)       =", rep["CR"]["f1-score"])
    print("Macro-F1     =", rep["macro avg"]["f1-score"])

if __name__ == "__main__":
    main()