#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_ultralytics_det_seeded_std.py

Runner générique pour modèles Ultralytics (YOLOv8/YOLOv11/RT-DETR, etc.)
avec reproductibilité "raisonnable" + sorties standardisées.

Remarques importantes :
  - Ultralytics ne peut pas être 100% déterministe sur CUDA (certains kernels,
    pré/post-traitements, etc.). On fixe néanmoins toutes les seeds et on désactive
    TF32 pour réduire la variance.
  - Pour produire un résultat "paper-ready", l'objectif est surtout :
      (i) split figé
      (ii) hyperparamètres versionnés
      (iii) un fichier pivot metrics.json

Exemples :
  # YOLOv8s
  python train_ultralytics_det_seeded_std.py \
    --model_path ".../yolov8s.pt" \
    --data_yaml  ".../data.yaml" \
    --runs_root  ".../runs_paper" \
    --dataset_id "arkeocera" --scenario "balanced" --split_id "splitA" \
    --model_id "yolov8s" --epochs 80 --batch 12 --imgsz 448 --seed 42 \
    --freeze_epochs 40

  # RT-DETR-L
  python train_ultralytics_det_seeded_std.py \
    --model_path ".../rtdetr-l.pt" \
    --data_yaml  ".../data.yaml" \
    --runs_root  ".../runs_paper" \
    --dataset_id "arkeocera" --scenario "balanced" --split_id "splitA" \
    --model_id "rtdetr-l" --epochs 80 --batch 12 --imgsz 448 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ultralytics import YOLO

from experiment_utils import (
    canonical_metrics_schema,
    make_run_dir,
    save_metrics_json,
    write_json,
    write_text,
)


def set_seed_all(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass


def purge_cache(path: str | Path) -> None:
    p = Path(path)
    if p.exists():
        try:
            p.unlink()
            print(f"[INFO] cache supprimé : {p}")
        except Exception as e:
            print(f"[WARN] purge_cache impossible: {e}")


def freeze_backbone(md: YOLO, freeze: bool = True) -> None:
    """Gèle/dégèle grossièrement les paramètres du backbone si présent."""
    try:
        for name, param in md.model.named_parameters():
            if "backbone" in name:
                param.requires_grad = not freeze
        for name, module in md.model.named_modules():
            if "backbone" in name:
                try:
                    module.eval() if freeze else module.train()
                except Exception:
                    pass
        print(f"[INFO] backbone {'gelé' if freeze else 'dégelé'}")
    except Exception as e:
        print(f"[WARN] freeze_backbone failed: {e}")


def results_to_metrics(results: Any) -> Dict[str, Any]:
    """Normalise (best-effort) les métriques Ultralytics dans un dict stable."""
    out: Dict[str, Any] = {}

    # results.results_dict existe dans la majorité des versions
    rd = getattr(results, "results_dict", None)
    if isinstance(rd, dict):
        out["results_dict"] = rd

        # Map fréquent (les clés peuvent varier selon versions)
        key_map = {
            "mAP50": [
                "metrics/mAP50(B)",
                "metrics/mAP50",  # parfois
                "metrics/mAP50-95(B)"  # fallback si bug
            ],
            "mAP50_95": [
                "metrics/mAP50-95(B)",
                "metrics/mAP50-95",
            ],
            "precision": ["metrics/precision(B)", "metrics/precision"],
            "recall": ["metrics/recall(B)", "metrics/recall"],
        }

        for k, candidates in key_map.items():
            for ck in candidates:
                if ck in rd:
                    out[k] = rd[ck]
                    break

    # speed info
    sp = getattr(results, "speed", None)
    if sp is not None:
        out["speed"] = sp

    # save_dir
    sd = getattr(results, "save_dir", None)
    if sd is not None:
        out["save_dir"] = str(sd)

    return out


def main() -> None:
    ap = argparse.ArgumentParser("Ultralytics detection runner (standardised)")
    ap.add_argument("--model_path", required=True, help=".pt / .yaml / modèle Ultralytics")
    ap.add_argument("--data_yaml", required=True, help="data.yaml (train/val/(test) + names)")
    ap.add_argument("--runs_root", required=True)

    ap.add_argument("--dataset_id", default="dataset")
    ap.add_argument("--scenario", default="default")
    ap.add_argument("--split_id", default=None)
    ap.add_argument("--model_id", default=None, help="ex: yolov8s / yolov11s / rtdetr-l")

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="'cuda', 'cpu' ou index GPU")

    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--lrf", type=float, default=0.10)
    ap.add_argument("--weight_decay", type=float, default=0.0005)
    ap.add_argument("--patience", type=int, default=50)

    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="AMP (désactivé par défaut)")
    ap.add_argument("--cache", default="disk", choices=["disk", "ram", "False", "true", "false"])
    ap.add_argument("--rect", action="store_true", help="rect=True (défaut: False)")

    ap.add_argument("--mosaic", type=float, default=0.0)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--augment", action="store_true", help="augment=True (défaut: False)")
    ap.add_argument("--plots", action="store_true", help="plots=True (défaut: False)")

    ap.add_argument("--val_during_train", action="store_true", help="val pendant train")
    ap.add_argument("--freeze_epochs", type=int, default=0, help="si >0: phase gelée puis resume")
    ap.add_argument("--eval_split", default="val", choices=["val", "test"], help="split final")
    ap.add_argument("--val_cache", default=None, help="chemin vers labels/val.cache (optionnel)")

    args = ap.parse_args()

    set_seed_all(args.seed)

    # Identifiants
    model_id = args.model_id or Path(args.model_path).stem
    paths = make_run_dir(
        runs_root=args.runs_root,
        dataset_id=args.dataset_id,
        scenario=args.scenario,
        task="det",
        model_id=model_id,
        run_id=None,
    )

    # Ultralytics écrit dans project/name => on force project=parent, name=run_dir.name
    project = str(paths.run_dir.parent)
    name = paths.run_dir.name

    # Cache param
    cache_val: Any
    if args.cache.lower() in {"false", "0", "none"}:
        cache_val = False
    elif args.cache.lower() in {"true", "1"}:
        cache_val = True
    else:
        cache_val = args.cache  # 'disk' ou 'ram'

    common_args: Dict[str, Any] = dict(
        data=args.data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        patience=args.patience,
        amp=bool(args.amp),
        cache=cache_val,
        rect=bool(args.rect),
        mosaic=args.mosaic,
        mixup=args.mixup,
        augment=bool(args.augment),
        plots=bool(args.plots),
        project=project,
        name=name,
        exist_ok=True,
        seed=args.seed,
    )

    if args.device is not None:
        common_args["device"] = args.device

    # On limite les augmentations pour comparaison contrôlée (si augment=False)
    if not args.augment:
        common_args.update(
            dict(
                hsv_h=0.005,
                hsv_s=0.5,
                hsv_v=0.3,
                degrees=0.0,
                translate=0.05,
                scale=0.10,
                shear=0.0,
                flipud=0.0,
                fliplr=0.5,
                auto_augment="none",
            )
        )

    # val pendant train ? (défaut : False -> on garde proche de tes scripts V3)
    common_args["val"] = bool(args.val_during_train)

    # Train
    print("[INFO] Chargement modèle:", args.model_path)
    model = YOLO(args.model_path)

    t0 = time.time()
    if args.freeze_epochs and 0 < args.freeze_epochs < args.epochs:
        # phase A
        print(f"[INFO] Phase A: freeze {args.freeze_epochs} epochs")
        freeze_backbone(model, True)
        model.train(epochs=args.freeze_epochs, **common_args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # phase B
        remaining = args.epochs - args.freeze_epochs
        if remaining > 0:
            print(f"[INFO] Phase B: unfreeze + resume ({remaining} epochs)")
            freeze_backbone(model, False)
            model.train(epochs=remaining, resume=True, **common_args)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    else:
        model.train(**common_args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    train_s = time.time() - t0

    # Eval finale
    if args.val_cache is not None:
        purge_cache(args.val_cache)

    print(f"[INFO] Eval finale split={args.eval_split}")
    results = model.val(
        data=args.data_yaml,
        imgsz=(args.imgsz, args.imgsz),
        rect=bool(args.rect),
        workers=args.workers,
        save_json=True,
        plots=bool(args.plots),
        split=args.eval_split,
    )

    metrics_std = results_to_metrics(results)
    write_json(paths.raw_dir / "ultralytics_metrics_raw.json", metrics_std)

    # metrics.json (pivot)
    hparams = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "amp": bool(args.amp),
        "cache": args.cache,
        "rect": bool(args.rect),
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "augment": bool(args.augment),
        "val_during_train": bool(args.val_during_train),
        "freeze_epochs": int(args.freeze_epochs),
        "device": args.device,
        "eval_split": args.eval_split,
        "train_time_sec": train_s,
    }
    payload = canonical_metrics_schema(
        task="det",
        model_id=model_id,
        dataset_id=args.dataset_id,
        scenario=args.scenario,
        split_id=args.split_id,
        split_hash=None,
        seed=args.seed,
        hparams=hparams,
        metrics={"final": metrics_std},
        paths=paths,
    )
    mj = save_metrics_json(paths, payload)
    write_text(paths.logs_dir / "DONE.txt", f"done\nmetrics_json={mj}\n")
    print(f"[OK] metrics.json: {mj}")


if __name__ == "__main__":
    main()
