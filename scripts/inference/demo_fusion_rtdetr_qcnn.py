#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate 3 images from one input image:
(a) RT-DETR detections only
(b) QCNN classification on RT-DETR crops
(c) post-hoc fusion result

Outputs in out_dir:
- <stem>_a_rtdetr_only.jpg
- <stem>_b_qcnn_on_crops.jpg
- <stem>_c_fusion.jpg
- per_image_json/<stem>.json
- fusion_demo.csv

If there is only ONE input image, the script also writes:
- a_rtdetr_only.jpg
- b_qcnn_on_crops.jpg
- c_fusion.jpg
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ultralytics import YOLO
import pennylane as qml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Internal class ids: 0=sigillee, 1=CO, 2=CR
DISPLAY_NAMES = {
    0: "terra sigillata",
    1: "OW",
    2: "RW",
}


def pil_open_rgb(p: Path) -> Image.Image:
    im = Image.open(p)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    ex = np.exp(x)
    return ex / ex.sum()


def pad_box(x1, y1, x2, y2, W, H, pad):
    bw = x2 - x1
    bh = y2 - y1
    px = bw * pad
    py = bh * pad
    x1 = max(0, int(x1 - px))
    y1 = max(0, int(y1 - py))
    x2 = min(W - 1, int(x2 + px))
    y2 = min(H - 1, int(y2 + py))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    den = area_a + area_b - inter
    return float(inter / den) if den > 0 else 0.0


def nms_greedy(boxes, scores, thr):
    idxs = list(range(len(scores)))
    idxs.sort(key=lambda i: scores[i], reverse=True)
    keep = []
    for i in idxs:
        ok = True
        for j in keep:
            if iou_xyxy(boxes[i], boxes[j]) >= thr:
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


def get_font(size=38):
    # Arial is often absent; DejaVuSans is usually available.
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_box(im: Image.Image, box, label: str, color=(255, 0, 0), width=4):
    draw = ImageDraw.Draw(im)
    font = get_font(38)
    x1, y1, x2, y2 = box

    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    pad = 8
    spacing = 6

    try:
        bb = draw.multiline_textbbox((0, 0), label, font=font, spacing=spacing)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        lines = label.split("\n")
        widths = []
        heights = []
        for line in lines:
            bb = draw.textbbox((0, 0), line, font=font)
            widths.append(bb[2] - bb[0])
            heights.append(bb[3] - bb[1])
        tw = max(widths) if widths else 0
        th = sum(heights) + max(0, len(lines) - 1) * spacing

    x_bg = max(0, x1)
    y_bg = max(0, y1 - th - 2 * pad - 4)
    x_bg2 = min(im.width - 1, x_bg + tw + 2 * pad)
    y_bg2 = min(im.height - 1, y_bg + th + 2 * pad)

    draw.rectangle([x_bg, y_bg, x_bg2, y_bg2], fill=color)
    draw.multiline_text(
        (x_bg + pad, y_bg + pad),
        label,
        fill=(255, 255, 255),
        font=font,
        spacing=spacing,
    )


def load_qcnn_weights(model, path: Path):
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        raise ValueError("Invalid QCNN checkpoint")
    model.load_state_dict(state_dict, strict=False)
    return model


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=6, n_layers=2):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.qparams = nn.Parameter(
            torch.zeros(self.n_layers, self.n_qubits, 3, dtype=torch.float64)
        )

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, angles_batch: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(angles_batch.shape[0]):
            x_cpu = angles_batch[i].to("cpu", dtype=torch.float64)
            out_list = self.circuit(x_cpu, self.qparams)
            out_t = torch.stack(list(out_list), dim=0)
            outs.append(out_t.to(device=angles_batch.device, dtype=angles_batch.dtype))
        return torch.stack(outs, dim=0)


class QCNN(nn.Module):
    def __init__(self, n_classes=3, n_qubits=6, n_layers=2, img_size=64):
        super().__init__()
        self.n_qubits = int(n_qubits)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            feat_dim = self.cnn(torch.zeros(1, 3, img_size, img_size)).shape[1]

        self.fc_to_qubits = nn.Linear(feat_dim, self.n_qubits)
        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(self.n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        f = self.cnn(x)
        angles = torch.tanh(self.fc_to_qubits(f)) * (math.pi / 2)
        q = self.quantum(angles)
        return self.classifier(q)


@torch.no_grad()
def qcnn_predict(model, tfm, crop, device):
    x = tfm(crop).unsqueeze(0).to(device)
    logits = model(x).detach().cpu().numpy()[0]
    probs = softmax_np(logits)
    cid = int(probs.argmax())
    return cid, float(probs[cid]), probs


def build_output_paths(out_dir: Path, stem: str, single_image: bool):
    paths = {
        "a": out_dir / f"{stem}_a_rtdetr_only.jpg",
        "b": out_dir / f"{stem}_b_qcnn_on_crops.jpg",
        "c": out_dir / f"{stem}_c_fusion.jpg",
    }

    if single_image:
        paths["a_simple"] = out_dir / "a_rtdetr_only.jpg"
        paths["b_simple"] = out_dir / "b_qcnn_on_crops.jpg"
        paths["c_simple"] = out_dir / "c_fusion.jpg"

    return paths


def save_three_images(ann_a, ann_b, ann_c, out_paths):
    ann_a.save(out_paths["a"])
    ann_b.save(out_paths["b"])
    ann_c.save(out_paths["c"])

    if "a_simple" in out_paths:
        ann_a.save(out_paths["a_simple"])
        ann_b.save(out_paths["b_simple"])
        ann_c.save(out_paths["c_simple"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rtdetr_best_pt", required=True)
    ap.add_argument("--qcnn_weights_only", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--min_area", type=float, default=0.002)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--annot_topn", type=int, default=10)
    ap.add_argument("--pad", type=float, default=0.03)
    ap.add_argument("--q_thr", type=float, default=0.0)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--force_replace_if_diff", action="store_true")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "per_image_json"
    out_json.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        print(f"[ERREUR] img_dir introuvable : {img_dir}")
        return

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )

    print(f"[INFO] img_dir : {img_dir}")
    print(f"[INFO] out_dir : {out_dir}")
    print(f"[INFO] device  : {device}")

    det = YOLO(args.rtdetr_best_pt)

    qcnn = QCNN(n_classes=3, n_qubits=6, n_layers=2, img_size=64)
    qcnn = load_qcnn_weights(qcnn, Path(args.qcnn_weights_only))
    qcnn.to(device).eval()

    tfm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    img_paths = [p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    img_paths.sort()

    print(f"[INFO] images trouvées : {len(img_paths)}")
    if not img_paths:
        print(f"[ERREUR] Aucune image trouvée dans {img_dir}")
        print(f"[ERREUR] Extensions acceptées : {sorted(IMG_EXTS)}")
        return

    single_image = len(img_paths) == 1

    csv_path = out_dir / "fusion_demo.csv"
    rows = [[
        "image", "idx", "x1", "y1", "x2", "y2",
        "det_cls", "det_conf",
        "qcnn_cls", "qcnn_conf", "qcnn_prob_detcls",
        "final_cls", "final_conf", "replaced"
    ]]

    for p in img_paths:
        print(f"[PROCESS] {p.name}")

        try:
            im = pil_open_rgb(p)
            W, H = im.size

            ann_a = im.copy()
            ann_b = im.copy()
            ann_c = im.copy()

            out_paths = build_output_paths(out_dir, p.stem, single_image)

            # Save placeholders immediately so A/B/C always exist
            save_three_images(ann_a, ann_b, ann_c, out_paths)

            yolo_device = 0 if device.type == "cuda" else str(device)

            r = det.predict(
                source=np.array(im),
                imgsz=args.imgsz,
                conf=args.conf,
                max_det=args.max_det,
                verbose=False,
                device=yolo_device,
            )[0]

            boxes = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    boxes.append({
                        "xyxy": [x1, y1, x2, y2],
                        "conf": float(confs[i]),
                        "cls": int(clss[i]),
                    })

            min_area_px = args.min_area * (W * H)
            boxes = [
                b for b in boxes
                if (max(0, b["xyxy"][2] - b["xyxy"][0]) * max(0, b["xyxy"][3] - b["xyxy"][1])) >= min_area_px
            ]

            if args.nms_iou > 0 and len(boxes) > 0:
                bxy = [b["xyxy"] for b in boxes]
                sco = [b["conf"] for b in boxes]
                keep = nms_greedy(bxy, sco, args.nms_iou)
                boxes = [boxes[i] for i in keep]

            boxes.sort(key=lambda b: b["conf"], reverse=True)
            if args.topk > 0:
                boxes = boxes[:args.topk]

            per_image = {
                "image": p.name,
                "num_boxes_kept": len(boxes),
                "objects": []
            }

            for k, b in enumerate(boxes):
                try:
                    x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
                    crop_x1, crop_y1, crop_x2, crop_y2 = pad_box(x1, y1, x2, y2, W, H, args.pad)
                    crop = im.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                    q_cls, q_conf, q_probs = qcnn_predict(qcnn, tfm, crop, device)
                    det_cls = b["cls"]
                    det_conf = b["conf"]
                    q_prob_det = float(q_probs[det_cls]) if 0 <= det_cls < 3 else 0.0

                    if args.force_replace_if_diff:
                        replaced = bool(q_cls != det_cls)
                    else:
                        replaced = bool(
                            (q_conf >= args.q_thr)
                            and ((q_conf - q_prob_det) >= args.margin)
                            and (q_cls != det_cls)
                        )

                    final_cls = q_cls if replaced else det_cls
                    final_conf = q_conf if replaced else det_conf

                    per_image["objects"].append({
                        "idx": k,
                        "bbox": [x1, y1, x2, y2],
                        "crop_bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
                        "det": {"cls": det_cls, "conf": det_conf},
                        "qcnn": {"cls": q_cls, "conf": q_conf, "probs": q_probs.tolist()},
                        "decision": {
                            "final_cls": final_cls,
                            "final_conf": final_conf,
                            "replaced": replaced,
                        },
                    })
                except Exception as e:
                    print(f"[WARN] objet ignoré sur {p.name}: {e}")
                    continue

            objs_sorted = sorted(
                per_image["objects"],
                key=lambda o: o["decision"]["final_conf"],
                reverse=True
            )[:args.annot_topn]

            for o in objs_sorted:
                x1, y1, x2, y2 = o["bbox"]
                det_cls = o["det"]["cls"]
                det_conf = o["det"]["conf"]
                q_cls = o["qcnn"]["cls"]
                q_conf = o["qcnn"]["conf"]
                fin_cls = o["decision"]["final_cls"]
                fin_conf = o["decision"]["final_conf"]
                replaced = o["decision"]["replaced"]

                # (a) RT-DETR only
                lbl_a = f"{DISPLAY_NAMES[det_cls]} {det_conf:.2f}"
                draw_box(ann_a, (x1, y1, x2, y2), lbl_a, color=(220, 20, 60), width=7)

                # (b) QCNN on crops
                lbl_b = f"{DISPLAY_NAMES[q_cls]} {q_conf:.2f}"
                draw_box(ann_b, (x1, y1, x2, y2), lbl_b, color=(30, 144, 255), width=7)

                # (c) Fusion
                if replaced:
                    delta = q_conf - det_conf
                    delta_pct = (delta / det_conf) * 100 if det_conf > 0 else 0.0
                    sign = "+" if delta >= 0 else ""
                    lbl_c = (
                        f"RT-DETR: {DISPLAY_NAMES[det_cls]} {det_conf:.2f}\n"
                        f"QCNN: {DISPLAY_NAMES[q_cls]} {q_conf:.2f}\n"
                        f"Δ: {sign}{delta:.2f} ({sign}{delta_pct:.0f}%)"
                    )
                    draw_box(ann_c, (x1, y1, x2, y2), lbl_c, color=(34, 139, 34), width=7)
                else:
                    lbl_c = f"{DISPLAY_NAMES[fin_cls]} {fin_conf:.2f}"
                    draw_box(ann_c, (x1, y1, x2, y2), lbl_c, color=(255, 140, 0), width=7)

            (out_json / f"{p.stem}.json").write_text(
                json.dumps(per_image, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            save_three_images(ann_a, ann_b, ann_c, out_paths)

            for o in per_image["objects"]:
                x1, y1, x2, y2 = o["bbox"]
                det_cls = o["det"]["cls"]
                det_conf = o["det"]["conf"]
                q_cls = o["qcnn"]["cls"]
                q_conf = o["qcnn"]["conf"]
                q_prob_det = float(o["qcnn"]["probs"][det_cls]) if 0 <= det_cls < 3 else 0.0
                fin_cls = o["decision"]["final_cls"]
                fin_conf = o["decision"]["final_conf"]
                rep = o["decision"]["replaced"]

                rows.append([
                    p.name, o["idx"], x1, y1, x2, y2,
                    DISPLAY_NAMES[det_cls], f"{det_conf:.4f}",
                    DISPLAY_NAMES[q_cls], f"{q_conf:.4f}", f"{q_prob_det:.4f}",
                    DISPLAY_NAMES[fin_cls], f"{fin_conf:.4f}", int(rep),
                ])

            print(f"[OK] sauvegardé : {out_paths['a'].name}, {out_paths['b'].name}, {out_paths['c'].name}")

        except Exception as e:
            print(f"[WARN] erreur sur {p.name}: {e}")
            continue

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("[OK] json   :", out_json)
    print("[OK] csv    :", csv_path)
    print("[OK] images :", out_dir)


if __name__ == "__main__":
    main()