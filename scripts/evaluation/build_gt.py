import os
import json
from PIL import Image

# ========= CONFIG =========
IMAGES_DIR = r"C:\ARKEOCERA\arkeocera\IA_ARKEO\ceramique\dataset\dataset_equilibre\images\val"
LABELS_DIR = r"C:\ARKEOCERA\arkeocera\IA_ARKEO\ceramique\dataset\dataset_equilibre\labels\val"
OUTPUT_JSON = "ground_truth.json"

# ==========================
images = []
annotations = []

image_id = 0
ann_id = 0

for file in os.listdir(IMAGES_DIR):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGES_DIR, file)
    label_path = os.path.join(LABELS_DIR, os.path.splitext(file)[0] + ".txt")

    img = Image.open(img_path)
    W, H = img.size

    images.append({
        "id": image_id,
        "file_name": file,
        "width": W,
        "height": H
    })

    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls, xc, yc, w, h = map(float, parts)

                # YOLO -> COCO
                x = (xc - w / 2) * W
                y = (yc - h / 2) * H
                w = w * W
                h = h * H

                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(cls),
                    "bbox": [x, y, w, h]
                })

                ann_id += 1

    image_id += 1


coco = {
    "images": images,
    "annotations": annotations,
    "categories": [
        {"id": 0, "name": "sigillee"},
        {"id": 1, "name": "CO"},
        {"id": 2, "name": "CR"}
    ]
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(coco, f)

print(f"✅ ground_truth.json généré ({len(images)} images, {len(annotations)} annotations)")