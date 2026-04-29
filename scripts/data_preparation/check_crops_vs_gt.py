import argparse
from pathlib import Path
import re
from collections import Counter, defaultdict

# extensions images possibles
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def load_label_counts(lbl_dir: Path):
    per_image_counts = {}
    per_image_classes = {}
    total = 0
    for p in sorted(lbl_dir.glob("*.txt")):
        with p.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        total += len(lines)
        per_image_counts[p.stem] = len(lines)
        # classes présentes dans ce fichier
        cls_ids = []
        for ln in lines:
            parts = ln.split()
            try:
                cls_id = int(float(parts[0]))
                cls_ids.append(cls_id)
            except Exception:
                pass
        per_image_classes[p.stem] = Counter(cls_ids)
    return total, per_image_counts, per_image_classes

def list_crops(crops_root: Path):
    # traverse sous-dossiers par classe (sigillee/CO/CR) et liste tous les PNG/JPG
    files = []
    for p in crops_root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files

def guess_cls_from_name(name: str):
    """
    Essaie d'extraire une classe depuis le nom de fichier, soit via '_c{int}', soit via dossier parent (sigillee/CO/CR).
    Retourne un tuple (cls_id_or_none, cls_name_or_none).
    """
    m = re.search(r"_c(\d+)", name)
    if m:
        return int(m.group(1)), None
    return None, None

def check_split(images_dir: Path, labels_dir: Path, crops_dir: Path, id2name=None):
    # 1) Comptage labels
    tot_lbl, per_img_cnt, per_img_cls = load_label_counts(labels_dir)

    # 2) Comptage crops total
    crops = list_crops(crops_dir)
    tot_crops = len(crops)

    # 3) Comptage crops par image-stem
    per_img_crops = Counter()
    per_img_crops_cls_guess = defaultdict(Counter)
    for c in crops:
        stem = c.stem
        # Si le crop a été produit du fichier foo.jpg/png -> on prend tout avant le premier suffixe généré
        # Approche simple: on retrouve l'image d'origine via préfixe exact du stem avant '_rN' si présent
        src_stem = stem
        m = re.search(r"(.*)_r\d+(?:_c\d+)?$", stem)
        if m:
            src_stem = m.group(1)
        per_img_crops[src_stem] += 1

        cid, _ = guess_cls_from_name(stem)
        if cid is not None:
            per_img_crops_cls_guess[src_stem][cid] += 1

    # 4) Résumés
    print("\n=== Vérification CROPS vs GT ===")
    print(f"[GT]   Total d'objets (somme des lignes .txt) : {tot_lbl}")
    print(f"[CROP] Total d'images crops trouvées          : {tot_crops}")
    diff = tot_crops - tot_lbl
    print(f"[DIFF] crops - gt = {diff} (doit idéalement être ≈ 0)")

    # 5) Deltas par image
    bad = []
    for stem, n_gt in per_img_cnt.items():
        n_crops = per_img_crops.get(stem, 0)
        if n_gt != n_crops:
            bad.append((stem, n_gt, n_crops))
    if bad:
        print("\n[ALERTE] Images avec mismatch (#crops vs #GT) :")
        for stem, n_gt, n_crops in bad[:30]:
            print(f"  - {stem}: gt={n_gt}, crops={n_crops}")
        if len(bad) > 30:
            print(f"  ... {len(bad)-30} de plus")

    # 6) Cohérence de classes via nommage '_c{cls}'
    #    (facultatif, dépend du naming ; si absent, on saute)
    has_cls_hint = any(guess_cls_from_name(p.stem)[0] is not None for p in crops)
    if has_cls_hint:
        mismatch_cls = []
        for stem, guess_counter in per_img_crops_cls_guess.items():
            gt_counter = per_img_cls.get(stem, Counter())
            # on regarde si toutes les classes guessées existent dans le GT
            for cid_guess, n_guess in guess_counter.items():
                if gt_counter[cid_guess] == 0:
                    mismatch_cls.append((stem, cid_guess, n_guess, dict(gt_counter)))
        if mismatch_cls:
            print("\n[ALERTE] Incohérences de classes (noms '_c{cls}') vs GT :")
            for stem, cidg, n, gt_cnt in mismatch_cls[:30]:
                print(f"  - {stem}: crops indiquent classe {cidg} ({n} fois) mais GT par image = {gt_cnt}")
            if len(mismatch_cls) > 30:
                print(f"  ... {len(mismatch_cls)-30} de plus")
    else:
        print("\n[INFO] Noms de crops sans suffixe '_c{cls}' -> test de classe par nom ignoré.")

    print("\n[OK] Vérification terminée.")

def main():
    ap = argparse.ArgumentParser(description="Vérifie que les crops correspondent bien aux GT YOLO.")
    ap.add_argument("--images", required=True, help="Dossier images du split (ex: .../images/train)")
    ap.add_argument("--labels", required=True, help="Dossier labels du split (ex: .../labels/train)")
    ap.add_argument("--crops",  required=True, help="Dossier crops_by_class (ex: .../crops_by_class/train)")
    args = ap.parse_args()

    images = Path(args.images)
    labels = Path(args.labels)
    crops  = Path(args.crops)
    assert images.exists(), f"Images introuvables: {images}"
    assert labels.exists(), f"Labels introuvables: {labels}"
    assert crops.exists(),  f"Crops introuvables : {crops}"

    check_split(images, labels, crops)

if __name__ == "__main__":
    main()
