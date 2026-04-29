#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prépare un dataset de crops redimensionnés (224x224) pour accélérer l'entraînement QCNN.
Entrée : ...\crops_dataset_equilibre\{train,val}\{classe}\*.png|jpg|...
Sortie : ...\crops_224\{train,val}\{classe}\*.png
"""
import os, shutil
from pathlib import Path
from PIL import Image

SRC = Path(r"C:\ARKEOCERA\arkeocera\IA_ARKEO\ceramique\crops_dataset_equilibre")
DST = Path(r"C:\ARKEOCERA\arkeocera\IA_ARKEO\ceramique\\crops_224")
SIZE = (224, 224)
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def main():
    for split in ("train","val"):
        for cls_dir in (SRC/split).iterdir():
            if not cls_dir.is_dir():
                continue
            out_dir = DST/split/cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for p in cls_dir.iterdir():
                if p.suffix.lower() not in IMG_EXTS: 
                    continue
                try:
                    with Image.open(p) as im:
                        im = im.convert("RGB")
                        im = im.resize(SIZE, Image.BILINEAR)
                        im.save(out_dir/p.name)  # PNG ou JPG selon extension d'origine
                except Exception as e:
                    print(f"[WARN] {p}: {e}")
    print(f"✅ Fini. Dataset prêt dans : {DST}")

if __name__ == "__main__":
    main()
