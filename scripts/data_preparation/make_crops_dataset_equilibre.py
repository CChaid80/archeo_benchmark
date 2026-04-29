#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Génère des crops par classe à partir des labels YOLO (GT) pour le dataset équilibré.
Crée deux splits : train/ et val/ sous 'crops_dataset_equilibre'.

Arborescence sortie :
  crops_dataset_equilibre/
    train/sigillee/..., train/CO/..., train/CR/...
    val/sigillee/...,   val/CO/...,   val/CR/...

Dépendances : opencv-python (cv2)
"""

import os
from pathlib import Path
import cv2
import json

# --- CONFIG PATHS (adapte si besoin) ---
DATA_ROOT   = Path(r"C:\ARKEOCERA\arkeocera\IA_ARKEO\ceramique\dataset\dataset_equilibre")
IMAGES_TRAIN = DATA_ROOT / "images" / "train"
LABELS_TRAIN = DATA_ROOT / "labels" / "train"
IMAGES_VAL   = DATA_ROOT / "images" / "val"
LABELS_VAL   = DATA_ROOT / "labels" / "val"

OUT_ROOT     = Path(r"C:\ARKEOCERA\arkeocera\IA_ARKEO\ceramique\crops_dataset_equilibre")

# --- CONFIG CLASSES (stable, sans accent dans les noms de dossiers) ---
ID2NAME = {
    0: "sigillee",
    1: "CO",
    2: "CR",
}
NAME2ID = {v: k for k, v in ID2NAME.items()}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def yolo_xywhn_to_xyxy(px, py, pw, ph, W, H):
    """YOLO (cx,cy,w,h) normalisé -> (x1,y1,x2,y2) en pixels + clip image."""
    cx, cy, w, h = px * W, py * H, pw * W, ph * H
    x1 = int(round(cx - w/2)); y1 = int(round(cy - h/2))
    x2 = int(round(cx + w/2)); y2 = int(round(cy + h/2))
    x1 = max(0, min(x1, W - 1)); y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1)); y2 = max(0, min(y2, H - 1))
    return x1, y1, x2, y2

def ensure_dirs(root: Path, split: str):
    """Crée les dossiers de sortie par classe pour un split donné (train/val)."""
    out_split = root / split
    out_split.mkdir(parents=True, exist_ok=True)
    dirs = {}
    for cid, name in ID2NAME.items():
        d = out_split / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[cid] = d
    return dirs

def export_split(img_dir: Path, lbl_dir: Path, out_root: Path, split: str, overwrite=False):
    assert img_dir.exists(), f"Images introuvables: {img_dir}"
    assert lbl_dir.exists(), f"Labels introuvables: {lbl_dir}"
    out_dirs = ensure_dirs(out_root, split)

    counts = {cid: 0 for cid in ID2NAME}
    unknown_cls = {}
    no_label = 0
    saved = 0

    img_paths = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT])
    print(f"\n[INFO] Split {split} | images: {len(img_paths)}")
    for img_path in img_paths:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Image illisible: {img_path}")
            continue
        H, W = img.shape[:2]

        if not lbl_path.exists():
            no_label += 1
            continue

        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for li, ln in enumerate(lines):
            parts = ln.split()
            if len(parts) < 5:
                print(f"[WARN] Ligne label invalide ({lbl_path}:{li}): {ln}")
                continue

            # classe
            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                print(f"[WARN] Classe non entière ({lbl_path}:{li}): {parts[0]}")
                continue

            # bbox
            try:
                px, py, pw, ph = map(float, parts[1:5])
            except ValueError:
                print(f"[WARN] Coords invalides ({lbl_path}:{li}): {' '.join(parts[1:])}")
                continue

            x1, y1, x2, y2 = yolo_xywhn_to_xyxy(px, py, pw, ph, W, H)
            if x2 <= x1 or y2 <= y1:
                # bbox dégénérée
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # dossier par classe
            if cls_id not in ID2NAME:
                unknown_cls[cls_id] = unknown_cls.get(cls_id, 0) + 1
                tgt_dir = out_root / split / f"unk_{cls_id}"
                tgt_dir.mkdir(parents=True, exist_ok=True)
            else:
                tgt_dir = out_dirs[cls_id]

            out_name = f"{img_path.stem}_r{li}_c{cls_id}.png"
            out_path = tgt_dir / out_name
            if overwrite or (not out_path.exists()):
                cv2.imwrite(str(out_path), crop)
                saved += 1
                if cls_id in counts:
                    counts[cls_id] += 1

    print(f"[SUMMARY {split}]")
    print(f" - images sans .txt : {no_label}")
    print(f" - total crops      : {saved}")
    for cid in sorted(counts.keys()):
        print(f"   * {cid} ({ID2NAME[cid]}): {counts[cid]}")
    if unknown_cls:
        print(" - classes inconnues :")
        for k, v in sorted(unknown_cls.items()):
            print(f"   * id={k} -> {v}")

    # petit rapport JSON
    report = {
        "split": split,
        "images_dir": str(img_dir),
        "labels_dir": str(lbl_dir),
        "out_root": str(out_root / split),
        "counts": {str(k): v for k, v in counts.items()},
        "unknown_cls": unknown_cls,
        "no_label_images": no_label,
        "id2name": ID2NAME,
    }
    (out_root / split).mkdir(parents=True, exist_ok=True)
    with open(out_root / split / "export_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✅ Rapport : {(out_root / split / 'export_report.json')}")

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    export_split(IMAGES_TRAIN, LABELS_TRAIN, OUT_ROOT, split="train", overwrite=False)
    export_split(IMAGES_VAL,   LABELS_VAL,   OUT_ROOT, split="val",   overwrite=False)
    print(f"\n✅ Crops exportés dans : {OUT_ROOT}")

if __name__ == "__main__":
    main()
