# Crop Pest Detection using YOLOv8-S

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-ffdd00.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-Academic-green.svg)]()

# Project Overview

This repository provides a complete YOLOv8-S pest detection pipeline for the AgroPest-12 dataset. It supports dataset preparation, model training, validation, testing, and automated result visualization, making it suitable for academic coursework and reproducible research. The system is modular, easy to extend, and optimized for GPU-based training.

# Key Features

- Full YOLOv8 workflow: training, validation, testing, inference
- Supports YOLOv5 / YOLOv8 / YOLOv11
- Automated YOLO dataset configuration (YAML generation)
- Automated results parsing: CSV → metrics → plots
- GPU-accelerated and configurable training pipeline
- Clean modular codebase with clear separation of functionality

# Project Structure

project/
├── dl_yolo.py                     # Main script for training/validation/testing
├── data_utils.py                  # Dataset utilities: download + YAML + checks
├── summarize_yolo_results.py      # Results summarization and visualization
├── count_dataset.py               # Dataset statistics tool
├── pest.yaml                      # YOLO dataset configuration
│
├── yolov8s.pt                     # YOLOv8-S pretrained weights
├── yolov8n.pt                     # YOLOv8-N pretrained weights
├── yolo11s.pt                     # YOLO11-S pretrained weights (optional)
│
└── runs_pest/                     # Auto-generated training output folders
    ├── yolov8s_tuned_v2/          # Training logs, best.pt, results.csv
    ├── yolov8s_tuned_v2_val/      # Validation predictions
    ├── yolov8s_tuned_v2_test/     # Test predictions
    └── ...

# File Descriptions

dl_yolo.py  
Main pipeline script for:
- --mode train — Train the YOLO model
- --mode val — Validate using the validation set
- --mode test — Test on the test set
- Logs, metrics, and model weights are saved under runs_pest/<run-name>/

Supports:
- YOLOv5 / YOLOv8 / YOLOv11
- Adjustable hyperparameters (epochs, lr, image size, mosaic, optimizer, batch size...)

data_utils.py  
Utility module responsible for:
- Downloading the AgroPest-12 dataset
- Decompressing and structuring YOLO-format folders
- Auto-generating pest.yaml
- Counting images/labels and verifying dataset integrity

summarize_yolo_results.py  
Reads YOLO’s built-in results.csv and generates:
- overall_metrics.csv
- per_class_metrics.csv
- mAP_summary.png
- precision_recall.png
- confusion_matrix.png

Saved inside:
runs_pest/<run-name>/

pest.yaml  
YOLO dataset configuration defining dataset paths and class names:
train: ./archive/train/images
val: ./archive/valid/images
test: ./archive/test/images
nc: 12
names: ['Ants','Bees','Beetles','Caterpillars','Earthworms','Earwigs','Grasshoppers','Moths','Slugs','Snails','Wasps','Weevils']

# Training the YOLOv8-S Model

Command:

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

Training outputs appear in:

runs_pest/yolov8s_tuned_v2/

Includes:
- weights/best.pt
- weights/last.pt
- results.csv
- runtime logs

# Validation

python dl_yolo.py --mode val \
  --run-name yolov8s_tuned_v2 \
  --device 0

Validation predictions saved under:
runs_pest/yolov8s_tuned_v2_val/

# Testing

python dl_yolo.py --mode test \
  --run-name yolov8s_tuned_v2 \
  --device 0

Test predictions saved under:
runs_pest/yolov8s_tuned_v2_test/

# Visualizing Results

python summarize_yolo_results.py --run-name yolov8s_tuned_v2

Generates:
- CSV summary files
- mAP bar charts
- Precision–Recall curves
- Confusion matrix
- Per-class visualizations

Saved inside:
runs_pest/yolov8s_tuned_v2/

# Modifying Configuration

Image size:
--imgsz 640 / 800 / 1024

Epochs:
--epochs 50 / 100 / 150 / 200

Batch size:
--batch 8 / 16 / 32

Optimizer:
--optimizer SGD | Adam | AdamW

Mosaic augmentation:
--mosaic 0.0 ~ 1.0

Dataset modification:
Edit pest.yaml to point to custom data.

# Citation

@software{yolov8_ultralytics,
  title={Ultralytics YOLOv8},
  author={Jocher, Glenn and contributors},
  year={2023},
  url={https://github.com/ultralytics/ultralytics},
}

# License

This project is intended for academic and research purposes only. Use of the dataset must comply with the original Kaggle license.
