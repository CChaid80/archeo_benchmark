import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD
# =========================
rtdetr = pd.read_csv("results_eval_rtdetr_seed42.csv")
yolo   = pd.read_csv("results_eval_yolov8_seed42.csv")

# =========================
# PLOT
# =========================
plt.figure(figsize=(8,6))

plt.plot(rtdetr["threshold"], rtdetr["macro_f1"], label="RT-DETR", linewidth=2)
plt.plot(yolo["threshold"], yolo["macro_f1"], label="YOLOv8", linewidth=2)

# point conf=0.25
plt.axvline(x=0.25, linestyle="--", color="gray")

plt.xlabel("Confidence threshold")
plt.ylabel("Macro-F1")
plt.title("Macro-F1 vs Confidence Threshold")
plt.legend()
plt.grid()

plt.savefig("figure12_corrected.png", dpi=300)
plt.show()