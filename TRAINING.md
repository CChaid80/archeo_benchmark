# Training scripts and launch commands

This repository includes the training scripts used for the archaeological
ceramic sherd benchmark. Datasets, trained weights and run outputs are not
stored in this repository.

## Important before training

Before launching any training script, set `CUBLAS_WORKSPACE_CONFIG` in the
same terminal session. This is required for deterministic CUDA execution when
the scripts enable deterministic PyTorch algorithms.

For `cmd.exe`:

```cmd
set CUBLAS_WORKSPACE_CONFIG=:4096:8
```

For PowerShell:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

## Scripts included

| Model family | Script |
| --- | --- |
| MobileNetV3-Small | `scripts/training/train_mobilenet_safe_seededV3.py` |
| ResNet-18 | `scripts/training/train_resnet18_desequilibre_paperready.py` |
| QCNN standard | `scripts/training/train_qcnn_safe_seededV2.py` |
| QCNN-VQE V2 standard | `scripts/training/train_qcnn_safe_seededV2_VQE_std.py` |
| YOLOv8s / YOLOv11s / RT-DETR-L | `scripts/training/train_ultralytics_det_seeded_std.py` |

## Local source files used for this commit

| Repository file | Local source |
| --- | --- |
| `scripts/training/train_mobilenet_safe_seededV3.py` | `Models_Finaux/train_mobilenet_safe_seededV3.py` |
| `scripts/training/train_resnet18_desequilibre_paperready.py` | `Models_Finaux/train_resnet18_desequilibre_paperready.py` |
| `scripts/training/train_qcnn_safe_seededV2.py` | `Models_Finaux/train_qcnn_safe_seededV2.py` |
| `scripts/training/train_qcnn_safe_seededV2_VQE_std.py` | `Models_Finaux/train_qcnn_safe_seededV2_VQE_std.py` |
| `scripts/training/train_ultralytics_det_seeded_std.py` | `Models_Finaux/train_ultralytics_det_seeded_std.py` |

## Dataset YAML files

| Scenario | Repository file | Local source |
| --- | --- | --- |
| Imbalanced detection dataset | `configs/datasets/arkeocera_imbalanced.yaml` | `Models_Finaux/data_imbalanced.yaml` |
| Balanced detection dataset | `configs/datasets/arkeocera_balanced.yaml` | `Models_Finaux/data_balanced.yaml` |

## Environment

Reference environment:

- Python >= 3.10
- PyTorch 2.5.1 + CUDA 12.1
- Ultralytics 8.3.225
- PennyLane 0.43.0

Install dependencies from:

```cmd
pip install -r requirements.txt
```

For CPU-only setup:

```cmd
pip install -r requirements-cpu.txt
```

For CUDA 12.1 PyTorch wheels, install PyTorch first with:

```cmd
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Reproducibility setup

Before launching the scripts in `cmd.exe`, set the reproducibility variable and
the project root:

```cmd
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PROJECT_ROOT=C:\ARKEOCERA\arkeocera\IA_ARKEO\ceramique
```

PowerShell equivalent:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
$env:PROJECT_ROOT="C:\ARKEOCERA\arkeocera\IA_ARKEO\ceramique"
```

The commands below use Windows `cmd.exe` syntax and assume they are launched
from the repository root.

## 1. MobileNetV3-Small

### Imbalanced dataset

```cmd
python scripts\training\train_mobilenet_safe_seededV3.py ^
  --train_dir "%PROJECT_ROOT%\runs\crops_desequilibre\train" ^
  --val_dir   "%PROJECT_ROOT%\runs\crops_desequilibre\val" ^
  --outdir    "%PROJECT_ROOT%\runs_paper\cls_mobilenet" ^
  --epochs 100 --batch_size 12 --img_size 640 --seed 42 --device cuda ^
  --imbalance_strategy none --loss ce
```

### Balanced dataset

```cmd
python scripts\training\train_mobilenet_safe_seededV3.py ^
  --train_dir "%PROJECT_ROOT%\crops_dataset_equilibre\train" ^
  --val_dir   "%PROJECT_ROOT%\crops_dataset_equilibre\val" ^
  --outdir    "%PROJECT_ROOT%\runs_paper\cls_mobilenet_equil" ^
  --epochs 100 --batch_size 12 --img_size 640 --seed 42 --device cuda ^
  --imbalance_strategy none --loss ce
```

## 2. ResNet-18

### Imbalanced dataset

```cmd
python scripts\training\train_resnet18_desequilibre_paperready.py ^
  --train_dir "%PROJECT_ROOT%\runs\crops_desequilibre\train" ^
  --val_dir   "%PROJECT_ROOT%\runs\crops_desequilibre\val" ^
  --runs_root "%PROJECT_ROOT%\runs_paper" ^
  --dataset_id "arkeocera" --scenario "imbalanced" --split_id "splitA" ^
  --epochs 80 --batch_size 12 --img_size 640 ^
  --lr 1e-4 --weight_decay 1e-4 ^
  --seed 42 --device cuda ^
  --class_weights none
```

### Balanced dataset

```cmd
python scripts\training\train_resnet18_desequilibre_paperready.py ^
  --train_dir "%PROJECT_ROOT%\crops_dataset_equilibre\train" ^
  --val_dir   "%PROJECT_ROOT%\crops_dataset_equilibre\val" ^
  --runs_root "%PROJECT_ROOT%\runs_paper" ^
  --dataset_id "arkeocera" --scenario "balanced" --split_id "splitA" ^
  --epochs 80 --batch_size 12 --img_size 640 ^
  --lr 1e-4 --weight_decay 1e-4 ^
  --seed 42 --device cuda ^
  --class_weights none
```

## 3. QCNN standard

### Imbalanced dataset

```cmd
python scripts\training\train_qcnn_safe_seededV2.py ^
  --train_dir "%PROJECT_ROOT%\runs\crops_desequilibre\train" ^
  --val_dir   "%PROJECT_ROOT%\runs\crops_desequilibre\val" ^
  --outdir    "%PROJECT_ROOT%\runs_paper\cls_qcnn" ^
  --epochs 80 --batch_size 12 --img_size 64 --seed 42 --device cuda ^
  --n_qubits 6 --n_q_layers 2 ^
  --imbalance_strategy none --loss ce
```

### Balanced dataset

```cmd
python scripts\training\train_qcnn_safe_seededV2.py ^
  --train_dir "%PROJECT_ROOT%\crops_dataset_equilibre\train" ^
  --val_dir   "%PROJECT_ROOT%\crops_dataset_equilibre\val" ^
  --outdir    "%PROJECT_ROOT%\runs_paper\cls_qcnn" ^
  --epochs 80 --batch_size 12 --img_size 64 --seed 42 --device cuda ^
  --n_qubits 6 --n_q_layers 2 ^
  --imbalance_strategy none --loss ce
```

## 4. QCNN-VQE V2

### Imbalanced dataset

```cmd
python scripts\training\train_qcnn_safe_seededV2_VQE_std.py ^
  --train_dir "%PROJECT_ROOT%\runs\crops_desequilibre\train" ^
  --val_dir   "%PROJECT_ROOT%\runs\crops_desequilibre\val" ^
  --runs_root "%PROJECT_ROOT%\runs_paper" ^
  --dataset_id "arkeocera" --scenario "imbalanced" --split_id "splitA" ^
  --epochs 80 --batch_size 12 --img_size 448 ^
  --seed 42 --device cuda ^
  --n_qubits 6 --n_layers 2 --class_weights none
```

### Balanced dataset

```cmd
python scripts\training\train_qcnn_safe_seededV2_VQE_std.py ^
  --train_dir "%PROJECT_ROOT%\crops_dataset_equilibre\train" ^
  --val_dir   "%PROJECT_ROOT%\crops_dataset_equilibre\val" ^
  --runs_root "%PROJECT_ROOT%\runs_paper" ^
  --dataset_id "arkeocera" --scenario "balanced" --split_id "splitA" ^
  --epochs 80 --batch_size 12 --img_size 448 ^
  --seed 42 --device cuda ^
  --n_qubits 6 --n_layers 2 --class_weights none
```

## 5. Detection on full images

The detection experiments use one generic Ultralytics script launched for
three model families and two dataset scenarios.

### YOLOv8s - imbalanced dataset

```cmd
python scripts\training\train_ultralytics_det_seeded_std.py ^
  --model_path "%PROJECT_ROOT%\congres_Inrap\yolov8s.pt" ^
  --data_yaml  "configs\datasets\arkeocera_imbalanced.yaml" ^
  --runs_root  "%PROJECT_ROOT%\runs_paper" ^
  --dataset_id "arkeocera" --scenario "imbalanced" --split_id "splitA" ^
  --model_id "yolov8s" --epochs 80 --batch 12 --imgsz 448 --seed 42
```

### YOLOv8s - balanced dataset

```cmd
python scripts\training\train_ultralytics_det_seeded_std.py ^
  --model_path "%PROJECT_ROOT%\congres_Inrap\yolov8s.pt" ^
  --data_yaml  "configs\datasets\arkeocera_balanced.yaml" ^
  --runs_root  "%PROJECT_ROOT%\runs_paper" ^
  --dataset_id "arkeocera" --scenario "balanced" --split_id "splitA" ^
  --model_id "yolov8s" --epochs 80 --batch 12 --imgsz 448 --seed 42
```

### YOLOv11s - imbalanced dataset

```cmd
python scripts\training\train_ultralytics_det_seeded_std.py ^
  --model_path "%PROJECT_ROOT%\congres_Inrap\yolo11s.pt" ^
  --data_yaml  "configs\datasets\arkeocera_imbalanced.yaml" ^
  --runs_root  "%PROJECT_ROOT%\runs_paper" ^
  --dataset_id "arkeocera" --scenario "imbalanced" --split_id "splitA" ^
  --model_id "yolo11s" --epochs 80 --batch 12 --imgsz 448 --seed 42
```

### YOLOv11s - balanced dataset

```cmd
python scripts\training\train_ultralytics_det_seeded_std.py ^
  --model_path "%PROJECT_ROOT%\congres_Inrap\yolo11s.pt" ^
  --data_yaml  "configs\datasets\arkeocera_balanced.yaml" ^
  --runs_root  "%PROJECT_ROOT%\runs_paper" ^
  --dataset_id "arkeocera" --scenario "balanced" --split_id "splitA" ^
  --model_id "yolo11s" --epochs 80 --batch 12 --imgsz 448 --seed 42
```

### RT-DETR-L - imbalanced dataset

```cmd
python scripts\training\train_ultralytics_det_seeded_std.py ^
  --model_path "%PROJECT_ROOT%\congres_Inrap\rtdetr-l.pt" ^
  --data_yaml  "configs\datasets\arkeocera_imbalanced.yaml" ^
  --runs_root  "%PROJECT_ROOT%\runs_paper" ^
  --dataset_id "arkeocera" --scenario "imbalanced" --split_id "splitA" ^
  --model_id "rtdetr-l" --epochs 80 --batch 12 --imgsz 448 --seed 42
```

### RT-DETR-L - balanced dataset

```cmd
python scripts\training\train_ultralytics_det_seeded_std.py ^
  --model_path "%PROJECT_ROOT%\congres_Inrap\rtdetr-l.pt" ^
  --data_yaml  "configs\datasets\arkeocera_balanced.yaml" ^
  --runs_root  "%PROJECT_ROOT%\runs_paper" ^
  --dataset_id "arkeocera" --scenario "balanced" --split_id "splitA" ^
  --model_id "rtdetr-l" --epochs 80 --batch 12 --imgsz 448 --seed 42
```

All detection results should be evaluated using the unified protocol described in [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md).
