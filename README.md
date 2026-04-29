# Archeo Benchmark

Benchmark of classical and hybrid quantum-classical models for the visual classification of Gallo-Roman ceramic sherds.

This repository accompanies the article:

**"Benchmarking deep and hybrid quantum-classical models for Gallo-Roman ceramic sherd classification: a reproducible evaluation framework"**

Cyrille Chaidron [1]*, Hafsa Taiebi Imrani [2]

[1] UMR 7041 ArScAn & U.R. 4284 TrAme, University of Picardie Jules Verne, Amiens, France  
[2] Faculty of Science, Ibn Tofail University, Kenitra, Morocco  
*Corresponding author: Cyrille Chaidron  
Email: cyrille@arteka.tech

It provides the scripts used to reproduce the evaluation protocol, including classification, object detection, and hybrid CNN/PQC experiments.

---

# Project description

The typological and technological identification of ceramic sherds is a central task in archaeological analysis, yet it still relies largely on expert visual assessment. In Gallo-Roman ceramics, some technological groups can be distinguished visually through surface appearance, firing colour, and fabric-related features, making this domain a relevant test case for computer vision.

This study proposes a reproducible evaluation framework to compare two computer-vision paradigms applied to ceramic fragments: crop-based classification and end-to-end object detection. The protocol is applied to a corpus of Gallo-Roman ceramics mainly from Amiens (France) and evaluates representative architectures, including compact convolutional networks (ResNet-18, MobileNetV3-Small), recent detectors (YOLOv8-s, YOLOv11-s, RT-DETR-L), and hybrid quantum-classical models (QCNN, QCNN-VQE).

Results show that crop-based classification achieves the highest and most stable performance, whereas end-to-end detectors are more sensitive to class imbalance and operating-point selection. Terra sigillata is consistently the most recognisable category, while most residual errors arise from confusions between the two common-ware classes (OW and RW). More broadly, the same archaeological dataset yields substantially different conclusions depending on whether the task is formulated as conditioned crop classification or full end-to-end detection.

Beyond model comparison, this work establishes a controlled and reproducible evaluation framework for computer vision applied to archaeological ceramics, and shows that task formulation is itself a major methodological factor in the interpretation of AI performance in heritage science.

---

# Overview

The objective of this benchmark is to compare several families of machine learning models for the classification of archaeological ceramic fragments.

Two complementary paradigms are evaluated:

- **Conditioned crop classification**  
  Classification of isolated ceramic sherds.

- **End-to-end detection**  
  Detection and classification of sherds directly in full images.

The benchmark includes:

- CNN classifiers (MobileNetV3-Small, ResNet-18)
- Hybrid CNN / PQC models (QCNN, QCNN-VQE)
- Object detectors (YOLOv8-s, YOLOv11-s, RT-DETR-L)

The dataset corresponds to a corpus of **Gallo-Roman ceramics mainly from Amiens (France)**.

For the full metric definition and detector matching procedure, see [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md).

---

# Dataset access

The dataset is not distributed directly in this repository. Access is limited
to research projects and requires a request by email to the contact address
listed above. Requests should explain the scientific context and intended use
of the dataset.

See `dataset/README.md` for details.

---

# Repository structure

- `dataset/`: dataset access information.
- `EVALUATION_PROTOCOL.md`: metric definition and detector matching protocol.
- `TRAINING.md`: training commands and reproducibility notes.
- `requirements.txt`: reference CUDA-oriented Python environment.
- `requirements-cpu.txt`: CPU-oriented Python environment.
- `configs/datasets/`: dataset YAML configuration files.
- `scripts/data_preparation/`: crop checking, crop generation, and dataset preparation scripts.
- `scripts/evaluation/`: ground-truth export, prediction export, metric computation, and plotting scripts.
- `scripts/training/`: training scripts for classification and detection experiments.
- `scripts/inference/`: inference and fusion demo scripts.
- `doc/`: figures and tables associated with the submission.
