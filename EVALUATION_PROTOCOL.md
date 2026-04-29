# Evaluation protocol

This document describes the evaluation protocol used in the archaeological ceramic sherd benchmark.

The aim of this protocol is to ensure that all results reported in the associated manuscript are computed in a transparent, reproducible and comparable way across model families.

The benchmark compares two complementary task formulations:

1. crop-based classification, where the object is already localised;
2. end-to-end object detection, where the model must both localise and classify ceramic sherds in full images.

These two formulations answer different scientific questions and must not be interpreted as equivalent tasks.

---

## 1. General objective

The objective of the benchmark is to evaluate the ability of several computer-vision models to recognise three technological categories of Gallo-Roman ceramic sherds.

The evaluated models include:

- crop-based image classifiers;
- compact convolutional neural networks;
- hybrid quantum-classical classifiers;
- end-to-end object detectors.

The evaluation framework is designed to distinguish between:

- visual class discrimination under ideal localisation conditions;
- complete end-to-end performance under realistic image-level detection conditions.

This distinction is central to the interpretation of the results.

---

## 2. Class definitions and canonical mapping

All experiments use the same three-class taxonomy.

| Class ID | Canonical class name | Description |
| --- | --- | --- |
| 0 | `terra_sigillata` | Terra sigillata |
| 1 | `OW` | Oxidised common ware |
| 2 | `RW` | Reduced common ware |

Some intermediate scripts or historical annotation files may contain equivalent labels such as:

| Historical / local label | Canonical label |
| --- | --- |
| `sigillee`, `sigillée` | `terra_sigillata` |
| `CO` | `OW` |
| `CR` | `RW` |

For the final evaluation, all labels must be remapped to the canonical order:

```text
0 = terra_sigillata
1 = OW
2 = RW
```

No additional class is allowed in the final evaluation files.

---

## 3. Dataset splitting principle

The dataset is split at the image level before any crop extraction.

This is essential to avoid data leakage between training and validation sets.

The correct workflow is:

```text
full annotated images
        |
        |-- image-level train / validation split
        |
        |-- crop extraction from train images only
        |
        |-- crop extraction from validation images only
```

Validation crops must never originate from images used during training.

This applies to all crop-based classifiers, including CNN, MobileNetV3-Small, ResNet-18, QCNN and QCNN-VQE models.

---

## 4. Task formulation 1: crop-based classification

### 4.1 Definition

In the crop-based classification setting, each input image corresponds to a single ceramic sherd extracted from a ground-truth bounding box.

The model receives an already-localised object and predicts one of the three classes.

This setting evaluates the intrinsic ability of the classifier to discriminate ceramic categories independently of the localisation problem.

### 4.2 Crop generation

Crops are generated from ground-truth annotations, not from detector predictions.

This ensures that crop-based classification results measure classification performance only, without being affected by detection failures.

The crop generation procedure uses:

```text
image + ground-truth bounding box -> cropped object image
```

Each crop is stored in a class-specific folder:

```text
crops/
  train/
    terra_sigillata/
    OW/
    RW/
  val/
    terra_sigillata/
    OW/
    RW/
```

or, depending on local naming conventions:

```text
crops/
  train/
    sigillee/
    CO/
    CR/
  val/
    sigillee/
    CO/
    CR/
```

Before final evaluation, these class names must be remapped to the canonical order.

### 4.3 Crop resizing

For crop-based classifiers, crops are resized to the input size required by the model.

Typical values used in the benchmark are:

| Model family | Input size |
| --- | --- |
| QCNN | 64 × 64 |
| QCNN-VQE | 64 × 64 or model-specific size |
| ResNet-18 | 640 × 640 |
| MobileNetV3-Small | 640 × 640 |

When a preprocessing script creates resized crops, it must preserve the train/validation split and the class-folder structure.

### 4.4 Crop-based classification metrics

For crop-based classifiers, standard multiclass classification metrics are computed:

- per-class precision;
- per-class recall;
- per-class F1-score;
- macro-F1;
- confusion matrix, when available.

For each class `c`:

```text
Precision_c = TP_c / (TP_c + FP_c)

Recall_c = TP_c / (TP_c + FN_c)

F1_c = 2 × Precision_c × Recall_c / (Precision_c + Recall_c)
```

If a denominator is zero, the corresponding metric is set to zero.

Macro-F1 is computed as the unweighted mean of the three class-wise F1 scores:

```text
Macro-F1 = mean(F1_terra_sigillata, F1_OW, F1_RW)
```

Macro-F1 is preferred over accuracy because it gives equal importance to all classes, independently of class frequency.

### 4.5 Interpretation of crop-based classification

Crop-based classification answers the question:

```text
How well can the model classify an already-localised ceramic sherd?
```

It does not measure full archaeological image-analysis performance because localisation is assumed to be correct.

Therefore, crop-based classification results must not be directly compared to end-to-end detection results as if they were the same task.

---

## 5. Task formulation 2: end-to-end object detection

### 5.1 Definition

In the end-to-end detection setting, models operate directly on full images.

Each model must:

1. localise each visible ceramic sherd;
2. assign a class to each detected sherd;
3. provide a confidence score.

A prediction is represented as:

```json
{
  "file_name": "image_001.jpg",
  "category_id": 0,
  "bbox": [x, y, width, height],
  "score": 0.87
}
```

Bounding boxes use absolute pixel coordinates in COCO format:

```text
[x_min, y_min, width, height]
```

### 5.2 Detection models

The end-to-end detection protocol is applied to:

- YOLOv8-s;
- YOLOv11-s;
- RT-DETR-L.

All detection models are evaluated using the same post-processing assumptions and the same external metric computation script.

---

## 6. Ground-truth format for detection evaluation

Detection ground truth is exported to a COCO-like JSON file containing:

```json
{
  "images": [
    {
      "id": 0,
      "file_name": "image_001.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "bbox": [x, y, width, height]
    }
  ],
  "categories": [
    {"id": 0, "name": "terra_sigillata"},
    {"id": 1, "name": "OW"},
    {"id": 2, "name": "RW"}
  ]
}
```

If source annotations are stored in YOLO format, they are converted from normalised YOLO coordinates:

```text
class_id x_center y_center width height
```

to absolute COCO coordinates:

```text
[x_min, y_min, width, height]
```

using the image width and height.

The exported ground-truth JSON is used only for evaluation.

---

## 7. Prediction export for detection models

Predictions are exported to JSON using the following structure:

```json
[
  {
    "file_name": "image_001.jpg",
    "category_id": 0,
    "bbox": [x, y, width, height],
    "score": 0.87
  }
]
```

The prediction file must contain:

- the image file name;
- the predicted class ID;
- the predicted bounding box in COCO format;
- the confidence score.

For confidence-threshold sweeps, predictions should be exported with a very low confidence threshold, for example:

```text
conf = 0.001
```

The final confidence threshold is then applied inside the evaluation script.

This avoids re-running inference for every confidence threshold.

---

## 8. Unified detection macro-F1 protocol

End-to-end detection performance is evaluated with a unified macro-F1 protocol.

The default parameters are:

| Parameter | Value |
| --- | --- |
| IoU threshold | 0.50 |
| Main confidence threshold | 0.25 |
| Maximum detections per image | 300 |
| Matching strategy | one-to-one greedy matching |
| Matching criterion | class-agnostic IoU first |
| Class check | after geometric matching |

This protocol is designed to jointly penalise:

- missed objects;
- false detections;
- incorrect class assignments;
- localisation errors.

---

## 9. Detection matching procedure

For each image, the following steps are applied.

### Step 1: confidence filtering

Predictions with confidence score lower than the selected threshold are discarded.

At the main operating point:

```text
confidence threshold = 0.25
```

### Step 2: maximum detections

Remaining predictions are sorted by decreasing confidence score and limited to:

```text
max_det = 300
```

### Step 3: candidate IoU pairs

For every remaining prediction and every ground-truth object in the same image, the intersection over union is computed.

A prediction/ground-truth pair is considered a candidate if:

```text
IoU >= 0.50
```

### Step 4: class-agnostic greedy matching

Candidate pairs are sorted by decreasing IoU.

Greedy one-to-one matching is then performed:

1. the highest-IoU candidate pair is selected;
2. the corresponding prediction and ground-truth object are marked as used;
3. all other candidate pairs involving either of them are discarded;
4. the process continues until no valid candidate pair remains.

Confidence score is used only as a deterministic secondary tie-breaker when needed.

Importantly, matching is class-agnostic.

This means that a prediction is first matched to the closest ground-truth object geometrically. The class is checked only after this geometric matching step.

---

## 10. TP, FP and FN assignment

After one-to-one matching, each prediction and ground-truth object contributes to the confusion counts according to the following rules.

### 10.1 Correct match

If a prediction is matched to a ground-truth object and both have the same class:

```text
TP += 1 for that class
```

Example:

```text
prediction: terra_sigillata
ground truth: terra_sigillata
IoU >= 0.50

=> TP_terra_sigillata += 1
```

### 10.2 Wrong-class match

If a prediction is matched to a ground-truth object but the predicted class is incorrect:

```text
FP += 1 for the predicted class
FN += 1 for the ground-truth class
```

Example:

```text
prediction: OW
ground truth: RW
IoU >= 0.50

=> FP_OW += 1
=> FN_RW += 1
```

This rule explicitly penalises class confusion after successful localisation.

### 10.3 Unmatched prediction

If a prediction is not matched to any ground-truth object:

```text
FP += 1 for the predicted class
```

Example:

```text
prediction: RW
no matching ground-truth object with IoU >= 0.50

=> FP_RW += 1
```

### 10.4 Unmatched ground-truth object

If a ground-truth object is not matched by any prediction:

```text
FN += 1 for the ground-truth class
```

Example:

```text
ground truth: OW
no prediction matched with IoU >= 0.50

=> FN_OW += 1
```

---

## 11. Detection precision, recall and F1

For each class `c`, the following metrics are computed:

```text
Precision_c = TP_c / (TP_c + FP_c)

Recall_c = TP_c / (TP_c + FN_c)
```

The class-wise F1-score is then:

```text
F1_c = 2 × Precision_c × Recall_c / (Precision_c + Recall_c)
```

If the denominator is zero, the metric is set to zero.

The macro-F1 score is the unweighted mean of the three class-wise F1 scores:

```text
Macro-F1 = mean(F1_terra_sigillata, F1_OW, F1_RW)
```

Macro-F1 is used as the main end-to-end detection metric because it treats all ceramic categories equally, regardless of class imbalance.

---

## 12. Main detection operating point

The main reported detection result is computed at the fixed operating point:

```text
IoU threshold = 0.50
confidence threshold = 0.25
max_det = 300
```

This operating point is used to compare YOLOv8-s, YOLOv11-s and RT-DETR-L under the same conditions.

The main operating point is not necessarily the best possible confidence threshold for every model. It is used as a fixed benchmark point for reproducibility and comparability.

---

## 13. Confidence-threshold sweep

In addition to the fixed operating point, a confidence-threshold sweep is performed.

The default sweep configuration is:

```text
threshold_start = 0.01
threshold_end = 0.95
threshold_num = 30
```

For each confidence threshold:

1. predictions below the threshold are discarded;
2. the same IoU-based one-to-one matching protocol is applied;
3. class-wise F1 and macro-F1 are recomputed.

The sweep is used to analyse the sensitivity of each detector to confidence calibration.

A robust detector should maintain high macro-F1 over a relatively broad range of thresholds.

A detector with unstable confidence calibration may show a sharp peak or a strong dependency on threshold selection.

---

## 14. Relationship between macro-F1 and COCO mAP

The benchmark reports both COCO-style detection metrics and the unified macro-F1 metric.

These metrics capture different aspects of performance.

### 14.1 COCO mAP

COCO mAP evaluates detection performance across confidence thresholds and, for mAP@0.5:0.95, across multiple IoU thresholds.

It is useful for assessing general localisation and ranking quality.

Typical metrics include:

```text
mAP@0.5
mAP@0.5:0.95
```

### 14.2 Unified macro-F1

The unified macro-F1 metric is computed at a fixed operating point and directly measures the balance between precision and recall for each class.

It is more sensitive to:

- the selected confidence threshold;
- class imbalance;
- class-specific false positives;
- class-specific false negatives;
- class confusion after localisation.

Therefore, a model may obtain a reasonable mAP while showing lower macro-F1 at a fixed confidence threshold.

This is not contradictory. It indicates differences in operating-point calibration and class-level decision stability.

---

## 15. Recommended evaluation scripts

The main scripts associated with the evaluation protocol are:

```text
scripts/evaluation/build_gt.py
scripts/evaluation/export_predictions_yolov8.py
scripts/evaluation/eval_detection_macro_f1_unified.py
scripts/evaluation/eval_qcnn_vqe_correct.py
scripts/plots/plot_f1_curves.py
```

The unified detection evaluation should be performed with:

```bash
python scripts/evaluation/eval_detection_macro_f1_unified.py \
  --pred-path path/to/predictions.json \
  --gt-path path/to/ground_truth.json \
  --output-prefix results/model_name \
  --main-conf 0.25 \
  --iou-thresh 0.5 \
  --max-det 300 \
  --class-names terra_sigillata,OW,RW
```

On Windows `cmd.exe`, the same command can be written as:

```cmd
python scripts\evaluation\eval_detection_macro_f1_unified.py ^
  --pred-path path\to\predictions.json ^
  --gt-path path\to\ground_truth.json ^
  --output-prefix results\model_name ^
  --main-conf 0.25 ^
  --iou-thresh 0.5 ^
  --max-det 300 ^
  --class-names terra_sigillata,OW,RW
```

---

## 16. Expected output files

The unified evaluation script produces several output files.

For an output prefix such as:

```text
results/yolov8_seed42
```

the expected outputs are:

```text
results/yolov8_seed42_conf025.txt
results/yolov8_seed42_conf025.json
results/yolov8_seed42_sweep.csv
results/yolov8_seed42_best.json
```

### 16.1 Main result text file

The text file contains a human-readable summary of the main operating point:

```text
Confidence threshold
IoU threshold
Max detections per image
Macro-F1
Class names
Precision per class
Recall per class
F1 per class
TP
FP
FN
```

### 16.2 Main result JSON file

The JSON file stores the same information in machine-readable form.

This file is recommended for reproducibility and downstream analysis.

### 16.3 Sweep CSV file

The sweep CSV file contains:

```text
threshold, macro_f1, is_main_conf
```

It is used to plot macro-F1 as a function of the confidence threshold.

### 16.4 Best-threshold JSON file

The best-threshold JSON file stores:

```text
best_threshold
best_macro_f1
main_conf
main_macro_f1
```

This file is used for threshold-sensitivity analysis only.

The main comparison between models should remain based on the fixed operating point unless explicitly stated otherwise.

---

## 17. Figure generation

Confidence-threshold sweep curves are generated from the sweep CSV files.

The plotted curve shows:

```text
x-axis: confidence threshold
y-axis: macro-F1
```

The fixed operating point at:

```text
confidence = 0.25
```

should be marked on the plot.

This allows direct visual comparison between:

- the fixed benchmark operating point;
- the best threshold found by sweep;
- the stability of each detector across thresholds.

---

## 18. Reproducibility requirements

For reproducible evaluation, the following information should be recorded for each run:

```text
model name
model version or checkpoint path
dataset scenario
train/validation split identifier
random seed
image size
confidence threshold
IoU threshold
maximum detections per image
prediction JSON path
ground-truth JSON path
evaluation script version
output files
```

For training reproducibility, the following should also be recorded:

```text
Python version
PyTorch version
Ultralytics version
PennyLane version
CUDA version
GPU model
random seed
training epochs
batch size
learning rate
weight decay
image size
augmentation settings
```

---

## 19. Interpretation guidelines

The two task formulations must be interpreted separately.

### 19.1 Crop-based classification

Crop-based classification answers:

```text
How well can a model classify an already-localised ceramic sherd?
```

This setting removes the localisation problem and focuses only on visual discrimination between ceramic categories.

High crop-based classification performance does not imply equivalent performance on full images.

### 19.2 End-to-end detection

End-to-end detection answers:

```text
How well can a model detect and classify ceramic sherds directly in full archaeological images?
```

This setting includes:

- object localisation;
- class prediction;
- confidence calibration;
- false positives;
- missed detections;
- class confusion.

It is therefore closer to practical archaeological image-analysis workflows.

### 19.3 Why both settings are reported

Both settings are scientifically useful.

Crop-based classification isolates the classification problem.

End-to-end detection evaluates the complete image-analysis pipeline.

The difference between the two is itself an important methodological result of the benchmark.

---

## 20. Recommended reporting format

For crop-based classifiers, report:

```text
per-class precision
per-class recall
per-class F1
macro-F1
model size
training scenario
input image size
seed
```

For detectors, report:

```text
mAP@0.5
mAP@0.5:0.95
precision
recall
macro-F1 at confidence = 0.25
per-class F1 at confidence = 0.25
confidence-threshold sweep
model size
inference time when available
seed
```

The manuscript should clearly distinguish between:

```text
classification on ground-truth crops
```

and:

```text
end-to-end detection on full images
```

---

## 21. Summary of the evaluation logic

The complete evaluation logic can be summarised as follows:

```text
Crop-based classification:
    ground-truth crop -> class prediction -> multiclass F1 / macro-F1

End-to-end detection:
    full image -> predicted boxes + classes + scores
    predictions filtered by confidence
    one-to-one IoU matching with ground truth
    class correctness checked after matching
    TP / FP / FN accumulated per class
    per-class F1 and macro-F1 computed
```

This protocol ensures that all models are evaluated under a controlled and comparable framework while preserving the methodological distinction between classification and detection.
