Pest Detection and Classification with YOLO
Supports YOLOv5 / YOLOv8 / YOLOv11

Functions:
    - Automatically download the Kaggle dataset (kagglehub)
    - Automatically generate pest.yaml (in the project directory)
    - Training: --mode train (Records training time, adjusts optimizer/lr, etc.)
    - Validation: --mode val (Records validation time + prints P/R/F1/mAP)
    - Prediction: --mode predict (Only draws boxes to view the effect after selecting a model)

Dependencies:
    pip install "numpy==1.26.4"
    pip install ultralytics kagglehub pyyaml

data_utils.py
Utility functions:
1. Download the Crop Pests Dataset from Kaggle using kagglehub.
2. Automatically generate pest.yaml (in the same directory as this file and dl_yolo.py).
3. Class names are detected automatically, not hard-coded:
   - Priority 1: read from names in data.yaml / dataset.yaml
   - Priority 2: try classes.txt / obj.names
   - Priority 3: infer class_0, class_1, ... from label files if nothing found
