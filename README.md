# Archeo Benchmark

Benchmark of classical and hybrid quantum-classical models for the visual classification of Gallo-Roman ceramic sherds.

This repository accompanies the article:

**"Benchmarking deep and hybrid quantum–classical models for Gallo-Roman ceramic classification"**

It provides the scripts used to reproduce the evaluation protocol, including classification, object detection, and hybrid CNN/PQC experiments.

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

The dataset corresponds to a corpus of **Gallo-Roman ceramics from Amiens (France)**.

---

# Repository structure
"# archeo_benchmark" 
