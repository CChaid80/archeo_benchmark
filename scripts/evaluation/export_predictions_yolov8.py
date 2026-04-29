from ultralytics import YOLO
import json
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = r"C:\ARTEKA\IA_ARKEO\ceramique\runs_paper\arkeocera\balanced\det\yolov8s\20260329-172930_yolo8_seed42\weights\best.pt"
IMAGES_DIR = r"C:\ARTEKA\IA_ARKEO\ceramique\dataset\dataset_equilibre\images\val"

OUTPUT_JSON = "predictions_yolo8_bon_seed42.json"

CONF = 0.001  # ⚠️ important pour sweep

# =========================
model = YOLO(MODEL_PATH)

predictions = []

# ⚠️ TRI OBLIGATOIRE (alignement stable)
files = sorted(os.listdir(IMAGES_DIR))

for file in files:
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(IMAGES_DIR, file)

    results = model.predict(path, conf=CONF, verbose=False)[0]

    if results.boxes is None:
        continue

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box

        predictions.append({
            "file_name": file,   # 🔥 clé stable
            "category_id": int(cls),
            "bbox": [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1)
            ],
            "score": float(score)
        })

# =========================
# SAVE
# =========================
with open(OUTPUT_JSON, "w") as f:
    json.dump(predictions, f)

print(f"✅ Predictions sauvegardées : {OUTPUT_JSON}")