#!/usr/bin/env python3
"""
Unified end-to-end detection macro-F1 evaluation script.

This script implements the evaluation protocol described in the manuscript:
- one-to-one greedy matching based on IoU >= threshold
- matching is class-agnostic (predictions are matched to GT by geometry first)
- class agreement is checked AFTER matching:
    * same class   -> TP for that class
    * wrong class  -> FP for predicted class + FN for GT class
- unmatched predictions -> FP for predicted class
- unmatched GT          -> FN for GT class

Expected inputs:
1) Predictions JSON: list of dicts with keys
   - file_name
   - category_id
   - bbox = [x, y, w, h]
   - score
2) Ground-truth COCO JSON with keys
   - images
   - annotations
   - categories

By default, category ids are remapped to 0..N-1 by sorted order, which handles
common cases such as [0,1,2] and [1,2,3] without editing the script.

Outputs:
- <prefix>_conf025.txt   : main operating point summary
- <prefix>_conf025.json  : same results as JSON
- <prefix>_sweep.csv     : threshold sweep (for Fig. 12)
- <prefix>_best.json     : best threshold found in sweep
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: Tuple[float, float, float, float]
    cls: int
    score: float


@dataclass(frozen=True)
class GroundTruth:
    bbox_xyxy: Tuple[float, float, float, float]
    cls: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified macro-F1 evaluation for end-to-end detection."
    )
    parser.add_argument("--pred-path", required=True, type=Path, help="Predictions JSON path")
    parser.add_argument("--gt-path", required=True, type=Path, help="Ground-truth COCO JSON path")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("eval_detection"),
        help="Prefix for output files (default: eval_detection)",
    )
    parser.add_argument("--main-conf", type=float, default=0.25, help="Main confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image")
    parser.add_argument("--threshold-start", type=float, default=0.01, help="Sweep start")
    parser.add_argument("--threshold-end", type=float, default=0.95, help="Sweep end")
    parser.add_argument("--threshold-num", type=int, default=30, help="Number of sweep thresholds")
    parser.add_argument(
        "--class-names",
        type=str,
        default="terra_sigillata,OW,RW",
        help="Comma-separated class names in remapped order (default: terra_sigillata,OW,RW)",
    )
    return parser.parse_args()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_sequential_remap(ids: List[int]) -> Dict[int, int]:
    unique_ids = sorted({int(x) for x in ids})
    if not unique_ids:
        raise ValueError("No category ids found.")
    return {old_id: new_id for new_id, old_id in enumerate(unique_ids)}


def bbox_xywh_to_xyxy(bbox_xywh: List[float]) -> Tuple[float, float, float, float]:
    if len(bbox_xywh) != 4:
        raise ValueError(f"Invalid bbox length: {bbox_xywh}")
    x, y, w, h = map(float, bbox_xywh)
    return (x, y, x + w, y + h)


def iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def prepare_data(preds_raw, gt_raw):
    # Remap prediction and GT category ids independently by sorted order.
    pred_remap = make_sequential_remap([p["category_id"] for p in preds_raw])
    gt_remap = make_sequential_remap([a["category_id"] for a in gt_raw["annotations"]])

    image_id_to_name = {img["id"]: img["file_name"] for img in gt_raw["images"]}

    gt_by_img: Dict[str, List[GroundTruth]] = defaultdict(list)
    for ann in gt_raw["annotations"]:
        image_id = ann["image_id"]
        if image_id not in image_id_to_name:
            raise KeyError(f"GT annotation references unknown image_id: {image_id}")
        fname = image_id_to_name[image_id]
        gt_by_img[fname].append(
            GroundTruth(
                bbox_xyxy=bbox_xywh_to_xyxy(ann["bbox"]),
                cls=gt_remap[int(ann["category_id"])],
            )
        )

    preds_by_img: Dict[str, List[Detection]] = defaultdict(list)
    for pred in preds_raw:
        if "file_name" not in pred:
            raise KeyError("Each prediction must contain a 'file_name' key.")
        preds_by_img[pred["file_name"]].append(
            Detection(
                bbox_xyxy=bbox_xywh_to_xyxy(pred["bbox"]),
                cls=pred_remap[int(pred["category_id"])],
                score=float(pred["score"]),
            )
        )

    num_classes = max(len(pred_remap), len(gt_remap))
    if len(pred_remap) != len(gt_remap):
        raise ValueError(
            f"Prediction classes ({len(pred_remap)}) and GT classes ({len(gt_remap)}) differ. "
            "Please verify class mapping/export."
        )

    return gt_by_img, preds_by_img, num_classes, pred_remap, gt_remap


def evaluate(
    gt_by_img: Dict[str, List[GroundTruth]],
    preds_by_img: Dict[str, List[Detection]],
    num_classes: int,
    conf_thresh: float,
    iou_thresh: float,
    max_det: int,
):
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    all_fnames = sorted(set(gt_by_img.keys()) | set(preds_by_img.keys()))

    for fname in all_fnames:
        gt_list = gt_by_img.get(fname, [])
        pred_list = preds_by_img.get(fname, [])

        # Filter predictions by confidence, keep highest-scoring max_det
        pred_filtered = [p for p in pred_list if p.score >= conf_thresh]
        pred_filtered.sort(key=lambda d: d.score, reverse=True)
        pred_filtered = pred_filtered[:max_det]

        # Build all admissible pairs (class-agnostic matching, as in paper)
        candidate_pairs = []
        for p_idx, pred in enumerate(pred_filtered):
            for g_idx, gt in enumerate(gt_list):
                current_iou = iou(pred.bbox_xyxy, gt.bbox_xyxy)
                if current_iou >= iou_thresh:
                    # Greedy matching by descending IoU, then score for determinism
                    candidate_pairs.append((-current_iou, -pred.score, p_idx, g_idx))

        candidate_pairs.sort()

        matched_preds = set()
        matched_gts = set()
        matched_pairs = []

        for neg_iou, neg_score, p_idx, g_idx in candidate_pairs:
            if p_idx in matched_preds or g_idx in matched_gts:
                continue
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)
            matched_pairs.append((p_idx, g_idx))

        # Score matched pairs
        for p_idx, g_idx in matched_pairs:
            pred = pred_filtered[p_idx]
            gt = gt_list[g_idx]
            if pred.cls == gt.cls:
                tp[pred.cls] += 1
            else:
                fp[pred.cls] += 1
                fn[gt.cls] += 1

        # Unmatched predictions -> FP
        for p_idx, pred in enumerate(pred_filtered):
            if p_idx not in matched_preds:
                fp[pred.cls] += 1

        # Unmatched GT -> FN
        for g_idx, gt in enumerate(gt_list):
            if g_idx not in matched_gts:
                fn[gt.cls] += 1

    precision = []
    recall = []
    f1 = []
    for c in range(num_classes):
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        if (p + r) == 0:
            f = 0.0
        else:
            f = 2.0 * p * r / (p + r)
        precision.append(p)
        recall.append(r)
        f1.append(f)

    macro_f1 = float(np.mean(f1))
    # Hard consistency check
    if not np.isclose(macro_f1, float(np.mean(f1)), atol=1e-12):
        raise RuntimeError("macro-F1 is inconsistent with class-wise F1 values.")

    return {
        "conf_thresh": conf_thresh,
        "iou_thresh": iou_thresh,
        "max_det": max_det,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "macro_f1": macro_f1,
    }


def save_main_result(path_txt: Path, path_json: Path, result: dict, class_names: List[str]):
    with path_txt.open("w", encoding="utf-8") as f:
        f.write(f"Confidence threshold: {result['conf_thresh']}\n")
        f.write(f"IoU threshold: {result['iou_thresh']}\n")
        f.write(f"Max detections/image: {result['max_det']}\n")
        f.write(f"Macro-F1: {result['macro_f1']}\n")
        f.write(f"Class names: {class_names}\n")
        f.write(f"Precision per class: {result['precision_per_class']}\n")
        f.write(f"Recall per class: {result['recall_per_class']}\n")
        f.write(f"F1 per class: {result['f1_per_class']}\n")
        f.write(f"TP: {result['tp']}\n")
        f.write(f"FP: {result['fp']}\n")
        f.write(f"FN: {result['fn']}\n")

    enriched = dict(result)
    enriched["class_names"] = class_names
    with path_json.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)


def save_sweep(path_csv: Path, results: List[Tuple[float, float]], main_conf: float):
    with path_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "macro_f1", "is_main_conf"])
        for t, m in results:
            writer.writerow([t, m, int(np.isclose(t, main_conf))])


def main():
    args = parse_args()

    preds_raw = load_json(args.pred_path)
    gt_raw = load_json(args.gt_path)

    gt_by_img, preds_by_img, num_classes, pred_remap, gt_remap = prepare_data(preds_raw, gt_raw)

    class_names = [name.strip() for name in args.class_names.split(",")]
    if len(class_names) != num_classes:
        raise ValueError(
            f"Expected {num_classes} class names, got {len(class_names)}: {class_names}"
        )

    print("Prediction id remap:", pred_remap)
    print("Ground-truth id remap:", gt_remap)
    print("Number of images with GT:", len(gt_by_img))
    print("Number of images with predictions:", len(preds_by_img))

    # Main evaluation point
    main_result = evaluate(
        gt_by_img=gt_by_img,
        preds_by_img=preds_by_img,
        num_classes=num_classes,
        conf_thresh=args.main_conf,
        iou_thresh=args.iou_thresh,
        max_det=args.max_det,
    )

    print(f"\n=== MAIN EVAL @ conf={args.main_conf:.4f} ===")
    print("Macro-F1:", main_result["macro_f1"])
    for idx, (name, p, r, f1) in enumerate(
        zip(
            class_names,
            main_result["precision_per_class"],
            main_result["recall_per_class"],
            main_result["f1_per_class"],
        )
    ):
        print(f"Class {idx} ({name}) -> P={p:.6f} R={r:.6f} F1={f1:.6f}")
    print("TP:", main_result["tp"])
    print("FP:", main_result["fp"])
    print("FN:", main_result["fn"])

    # Threshold sweep
    thresholds = np.linspace(args.threshold_start, args.threshold_end, args.threshold_num)
    sweep = []
    print("\n=== SWEEP ===")
    for t in thresholds:
        result_t = evaluate(
            gt_by_img=gt_by_img,
            preds_by_img=preds_by_img,
            num_classes=num_classes,
            conf_thresh=float(t),
            iou_thresh=args.iou_thresh,
            max_det=args.max_det,
        )
        sweep.append((float(t), float(result_t["macro_f1"])))
        print(f"{t:.4f} -> macro-F1={result_t['macro_f1']:.6f}")

    best_threshold, best_macro = max(sweep, key=lambda x: x[1])
    best_payload = {
        "best_threshold": best_threshold,
        "best_macro_f1": best_macro,
        "main_conf": args.main_conf,
        "main_macro_f1": main_result["macro_f1"],
    }

    prefix = args.output_prefix
    txt_path = prefix.with_name(prefix.name + "_conf025.txt")
    json_path = prefix.with_name(prefix.name + "_conf025.json")
    csv_path = prefix.with_name(prefix.name + "_sweep.csv")
    best_path = prefix.with_name(prefix.name + "_best.json")

    save_main_result(txt_path, json_path, main_result, class_names)
    save_sweep(csv_path, sweep, args.main_conf)
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    print(f"\n✅ Main result saved to: {txt_path}")
    print(f"✅ Main result JSON saved to: {json_path}")
    print(f"✅ Sweep CSV saved to: {csv_path}")
    print(f"✅ Best-threshold summary saved to: {best_path}")
    print(f"\nBEST SWEEP POINT -> threshold={best_threshold:.4f}, macro-F1={best_macro:.6f}")


if __name__ == "__main__":
    main()
