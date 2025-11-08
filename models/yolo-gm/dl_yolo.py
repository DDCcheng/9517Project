import argparse  # For parsing command line arguments
import time      # To measure training time
from pathlib import Path  # For handling file paths

from ultralytics import YOLO  # Import YOLO model from ultralytics
from data_utils import get_or_create_pest_yaml  # Import custom function from data_utils.py

# Get the current directory
PROJECT_ROOT = Path(__file__).resolve().parent
# Directory to store all results
RUNS_ROOT = PROJECT_ROOT / "runs_pest"


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Pest Detection and Classification - Train / Validate / Predict"
    )

    # Basic arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "val", "predict"],
        help="Run mode: train / val / predict",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        help=(
            "Model name or .pt file path, e.g. yolov5n, yolov8n, yolo11n, "
            "or a trained model such as runs_pest/yolov8n_final/weights/best.pt"
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (short side), recommended to be a multiple of 32 (e.g., 320/640)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (used only in mode=train)",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (used only in mode=train)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: '0' for the first GPU; '0,1' for multi-GPU; 'cpu' to force CPU usage",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="exp",
        help="Experiment name, results will be saved under runs_pest/<run-name>",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of Dataloader workers",
    )

    # Additional hyperparameters (for tuning)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["SGD", "Adam", "AdamW", "RMSProp"],
        help="Optimizer type, AdamW is recommended for better convergence",
    )

    parser.add_argument(
        "--lr0",
        type=float,
        default=0.005,
        help="Initial learning rate, recommended default = 0.005",
    )

    parser.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Mosaic data augmentation probability (0~1), 1.0 means always enabled",
    )

    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation (enabled by default); useful for ablation studies",
    )

    # Prediction-related parameters
    parser.add_argument(
        "--source",
        type=str,
        default=str(PROJECT_ROOT),
        help="Input source for predict mode: image/file/folder/video path",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions (used only in mode=predict)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate: val (validation) or test (final evaluation).",
    )

    return parser.parse_args()

"""
Load YOLO model:
    - If provided a keyword such as 'yolov5n', 'yolov8n', 'yolo11n', automatically append '.pt'
    - If provided a full path like 'runs_pest/yolov8n_final/weights/best.pt', load directly
"""
def load_model(model_name: str) -> YOLO:
    model_name = model_name.strip()

    # If it's not a .pt or .yaml file, append .pt automatically (e.g., 'yolov8n' -> 'yolov8n.pt')
    if not (model_name.endswith(".pt") or model_name.endswith(".yaml")):
        model_name = model_name + ".pt"

    print(f"[dl_yolo] Loading model weights: {model_name}")
    model = YOLO(model_name)  # ultralytics automatically downloads weights if it's an official model name
    return model


def main():
    args = parse_args()

    print(f"[dl_yolo] Project root: {PROJECT_ROOT}")
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    # 1) Get or create pest.yaml (contains dataset paths and class names)
    data_yaml = get_or_create_pest_yaml(PROJECT_ROOT)
    print(f"[dl_yolo] Using dataset config: {data_yaml}")

    # 2) Load model
    model = load_model(args.model)

    # 3) Execute different logic according to mode
    if args.mode == "train":
        print("[dl_yolo] Starting training ...")
        t0 = time.time()
        results = model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(RUNS_ROOT),
            name=args.run_name,
            workers=args.workers,
            exist_ok=True, # Reuse existing run folder if it exists

            # Extra hyperparameters
            optimizer=args.optimizer,
            lr0=args.lr0,
            mosaic=args.mosaic,
            augment=(not args.no_augment),
        )
        t1 = time.time()
        total_time = t1 - t0

        print("[dl_yolo] Training complete. Main outputs in:", RUNS_ROOT / args.run_name)
        print(f"[summary] Training time: {total_time/60:.2f} min "
              f"(~{total_time/max(args.epochs,1):.2f} s/epoch)")

    elif args.mode == "val":
        print(f"[dl_yolo] Starting evaluation on {args.split} set ...")
        t0 = time.time()
        metrics = model.val(
            data=data_yaml,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(RUNS_ROOT),
            name=f"{args.run_name}_{args.split}",
            split=args.split,
            exist_ok=True,
        )
        t1 = time.time()
        total_time = t1 - t0

        # Print detection metrics + compute F1 + speed
        try:
            p = float(metrics.box.mp) # mean precision
            r = float(metrics.box.mr) # mean recall
            map50 = float(metrics.box.map50)
            map5095 = float(metrics.box.map)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            print("\n[summary] Validation metrics (YOLO box-level):")
            print(f"  Precision (mp):       {p:.5f}")
            print(f"  Recall    (mr):       {r:.5f}")
            print(f"  F1-score (2PR/(P+R)): {f1:.5f}")
            print(f"  mAP@0.5:              {map50:.5f}")
            print(f"  mAP@0.5:0.95:         {map5095:.5f}")
        except Exception as e:
            print("[dl_yolo] Failed to parse P/R/mAP from metrics object. "
                  "You can print(metrics) to check manually. Error:", e)

        try:
            inf_ms = float(metrics.speed.get("inference", 0.0))
            print(f"[summary] Validation speed: {inf_ms:.2f} ms / image")
        except Exception:
            print(f"[summary] Total validation time: {total_time:.2f} s")

    elif args.mode == "predict":
        print("[dl_yolo] Starting prediction ...")
        t0 = time.time()
        preds = model.predict(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            project=str(RUNS_ROOT),
            name=args.run_name + "_pred",
            save=True, # Save images with detection boxes
            exist_ok=True,
        )
        t1 = time.time()
        total_time = t1 - t0
        print("[dl_yolo] Prediction complete. Visual results saved to:",
              RUNS_ROOT / (args.run_name + "_pred"))
        print(f"[summary] Prediction total time: {total_time:.2f} s")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
