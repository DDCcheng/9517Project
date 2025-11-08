"""
Small tool: Reads the last line from runs_pest/<run-name>/results.csv,
outputs precision/recall/F1/mAP@0.5/mAP@0.5:0.95.
Usage:
    python summarize_yolo_results.py --run-name yolov8n_final
"""

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_PEST = PROJECT_ROOT / "runs_pest"

# Compatible with two directory structures: runs_pest/<run>/results.csv or runs_pest/detect/<run>/results.csv
if (RUNS_PEST / "detect").exists():
    RUNS_ROOT = RUNS_PEST / "detect"
else:
    RUNS_ROOT = RUNS_PEST


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-name",
        type=str,
        default="exp",
        help="The run name used during training (e.g., exp, yolov8n_final).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = RUNS_ROOT / args.run_name
    csv_path = run_dir / "results.csv"

    if not csv_path.exists():
        print(f"[error] Cannot find {csv_path}, please check if the run-name is correct.")
        return

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("[error] results.csv is empty.")
        return

    last = rows[-1]  # Last line = last epoch

    # Automatically detect precision / recall / mAP column names (YOLOv8 log field names)
    prec_col = next((c for c in last.keys() if "metrics/precision" in c), None)
    rec_col = next((c for c in last.keys() if "metrics/recall" in c), None)
    map50_col = next((c for c in last.keys()
                      if "metrics/mAP50" in c or "metrics/mAP_0.5" in c), None)
    map5095_col = next((c for c in last.keys()
                        if "metrics/mAP50-95" in c or "metrics/mAP_0.5:0.95" in c),
                       None)

    def get_float(col_name):
        if col_name is None:
            return None
        try:
            return float(last[col_name])
        except Exception:
            return None

    p = get_float(prec_col)
    r = get_float(rec_col)
    map50 = get_float(map50_col)
    map5095 = get_float(map5095_col)
    f1 = 2 * p * r / (p + r) if (p is not None and r is not None and (p + r) > 0) else None

    print(f"[run] {args.run_name}")
    print("  from:", csv_path)
    print("  detected columns:")
    print("   precision:", prec_col)
    print("   recall:   ", rec_col)
    print("   mAP@0.5:  ", map50_col)
    print("   mAP@0.5:0.95:", map5095_col)

    print("run_name, precision, recall, F1, mAP50, mAP50-95")
    print(f"{args.run_name}, "
          f"{p if p is not None else 'NA'}, "
          f"{r if r is not None else 'NA'}, "
          f"{f1 if f1 is not None else 'NA'}, "
          f"{map50 if map50 is not None else 'NA'}, "
          f"{map5095 if map5095 is not None else 'NA'}")


if __name__ == "__main__":
    main()
