# 👁️ COMP9517 — Group Project

## 🧩 Group Name

**EyeCrew**

# Crop Pest Detection on AgroPest-12 (Faster R-CNN & YOLOv8-S)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
 [![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
 [![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-ffdd00.svg)](https://github.com/ultralytics/ultralytics)
 [![License](https://img.shields.io/badge/license-Academic-green.svg)](https://chatgpt.com/c/691c2399-b94c-8323-a627-2542068ff51c)

------

## Project Overview

This repository implements **two independent object detection pipelines** for the **AgroPest-12** crop pest dataset:

1. **Faster R-CNN (ResNet50-FPN, PyTorch)**
2. **YOLOv8-S (Ultralytics)**

Both models perform **multi-class pest detection** with bounding boxes over 12 classes (ants, beetles, caterpillars, etc.), and support **end-to-end training, evaluation, and automated result visualization** for academic coursework and reproducible research.

The overall workflow is:

> Dataset preparation → Training → Evaluation → Metrics & plots → (Optional) Model comparison

------

## Models Implemented

| Model        | Framework   | Use Case                             |
| ------------ | ----------- | ------------------------------------ |
| Faster R-CNN | PyTorch     | Strong baseline, flexible evaluation |
| YOLOv8-S     | Ultralytics | Fast, production-style detector      |

Each model has its own scripts and output folders, but they share the same **AgroPest-12** dataset and class definitions.

------

## Dataset: AgroPest-12

- **Classes (12):**

  | ID   | Pest Name    |
  | ---- | ------------ |
  | 0    | Ants         |
  | 1    | Bees         |
  | 2    | Beetles      |
  | 3    | Caterpillars |
  | 4    | Earthworms   |
  | 5    | Earwigs      |
  | 6    | Grasshoppers |
  | 7    | Moths        |
  | 8    | Slugs        |
  | 9    | Snails       |
  | 10   | Wasps        |
  | 11   | Weevils      |

- **Format:** YOLO-style labels (`.txt`) with `class cx cy w h`

- **Split:** `train/`, `valid/`, `test/` (each with `images/` and `labels/`)

- Dataset paths and names are configured via:

  - `dataset/data.yaml` for **Faster R-CNN** 
  - `pest.yaml` for **YOLOv8-S** 

------

## Repository Structure (Example)

```bash
project_root/
├── FASTER_R_CNN_JARED/          # Faster R-CNN pipeline (PyTorch)
│   ├── faster_r_cnn_marked.py
│   ├── evaluate_best.py
│   ├── map_evaluate_valid.py
│   ├── map_evaluate_test.py
│   ├── prepare.sh
│   ├── train.sh
│   ├── dataset/
│   │   ├── data.yaml
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── fast_r_cnn/
│       ├── eval_out_frcnn_valid/
│       ├── eval_out_frcnn_test/
│       └── runs_frcnn/
│
├── dl_yolo.py                   # YOLOv8-S pipeline entry script
├── data_utils.py                # Dataset download + YAML + checks
├── summarize_yolo_results.py    # YOLO result summarization & plots
├── count_dataset.py             # Dataset statistics
├── pest.yaml                    # YOLO dataset configuration
└── runs_pest/                   # YOLO training/eval outputs
    ├── yolov8s_tuned_v2/
    ├── yolov8s_tuned_v2_val/
    └── yolov8s_tuned_v2_test/
```

> 你可以根据自己实际的文件夹名字稍微改一下上面的结构说明即可。

------

## Environment Setup

You can either create a **single Conda environment** and install both PyTorch and Ultralytics, or follow the original per-model scripts.

### Option A — Use the provided scripts (per-model)

**Faster R-CNN:**

```bash
cd FASTER_R_CNN_JARED
bash prepare.sh        # dataset + environment + basic deps
# or
bash train.sh          # create env, install deps, run training
```



**YOLOv8-S:**

```bash
# In project_root
# Typical manual install:
pip install ultralytics opencv-python matplotlib numpy<2
```



### Option B — Single unified environment (example)

```bash
conda create -n pestdet python=3.10 -y
conda activate pestdet

# PyTorch with CUDA 12.1 (modify if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Common deps
pip install ultralytics opencv-python tqdm matplotlib scikit-learn numpy<2
```

------

## 1. Faster R-CNN Pipeline (PyTorch)

### Training

From `FASTER_R_CNN_JARED/`:

```bash
python faster_r_cnn_marked.py \
  --data_root ./dataset \
  --epochs 20 \
  --batch_size 4 \
  --lr 1e-3 \
  --num_classes 12 \
  --eval_map \
  --eval_prf1 \
  --eval_prauc
```

- Best model checkpoint saved to: `fast_r_cnn/runs_frcnn/best.pth`
- Training logs: `train_output.log` 

### Evaluation

**Validation:**

```bash
python map_evaluate_valid.py \
  --weights fast_r_cnn/runs_frcnn/best.pth \
  --data dataset/data.yaml \
  --output fast_r_cnn/eval_out_frcnn_valid
```

**Test:**

```bash
python map_evaluate_test.py \
  --weights fast_r_cnn/runs_frcnn/best.pth \
  --data dataset/data.yaml \
  --output fast_r_cnn/eval_out_frcnn_test
```

**Summary report:**

```bash
python evaluate_best.py
```

This exports:

- `metrics.json`
- `overall_metrics.csv`, `per_class_metrics.csv`
- `overall_mAP_summary.png`, `per_class_mAP_bar.png`
- ROC / PR curve plots 

------

## 2. YOLOv8-S Pipeline (Ultralytics)

### Training

From `project_root`:

```bash
python dl_yolo.py --mode train \
  --model yolov8s.pt \
  --epochs 150 \
  --imgsz 800 \
  --batch 16 \
  --device 0 \
  --run-name yolov8s_tuned_v2 \
  --optimizer AdamW \
  --lr0 0.003 \
  --mosaic 0.8
```

Outputs go to:

```bash
runs_pest/yolov8s_tuned_v2/
# includes: weights/best.pt, weights/last.pt, results.csv, logs...
```



### Validation & Testing

```bash
# Validation
python dl_yolo.py --mode val \
  --run-name yolov8s_tuned_v2 \
  --device 0

# Testing
python dl_yolo.py --mode test \
  --run-name yolov8s_tuned_v2 \
  --device 0
```

Predictions saved to:

- `runs_pest/yolov8s_tuned_v2_val/`
- `runs_pest/yolov8s_tuned_v2_test/` 

### Result Summarization

```bash
python summarize_yolo_results.py --run-name yolov8s_tuned_v2
```

Generates:

- `overall_metrics.csv`, `per_class_metrics.csv`
- `mAP_summary.png`
- `precision_recall.png`
- `confusion_matrix.png`
- Per-class visualizations

All saved under `runs_pest/yolov8s_tuned_v2/`. 

------

## Evaluation Metrics (Both Pipelines)

Typical metrics reported:

- **mAP@50 / mAP@75**
- **mAR@100**
- **Precision / Recall / F1**
- **Accuracy (image-level)**
- **ROC-AUC / PR-AUC**
- (YOLO only) confusion matrix, per-class curves and bars

You can directly compare Faster R-CNN vs YOLOv8-S using their CSV and PNG outputs to discuss trade-offs between accuracy and speed in your report.

------

## Citations

**Faster R-CNN:**

```bibtex
@article{ren2015faster,
  title     = {Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
  author    = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year      = {2017},
  volume    = {39},
  number    = {6},
  pages     = {1137--1149},
  doi       = {10.1109/TPAMI.2016.2577031},
  url       = {https://arxiv.org/abs/1506.01497}
}
```



**YOLOv8:**

```bibtex
@software{yolov8_ultralytics,
  title={Ultralytics YOLOv8},
  author={Jocher, Glenn and contributors},
  year={2023},
  url={https://github.com/ultralytics/ultralytics},
}
```



------

## License

This project is intended for **academic and research purposes only**.
 Use of the AgroPest-12 dataset must comply with the original Kaggle license.
