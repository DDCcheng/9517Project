#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Faster R-CNN (best.pth) on a YOLO-format detection set and export:
  • Detection: mAP, mAP@50, mAP@75, mAR@100 (torchmetrics)
  • Micro P/R/F1/Accuracy @ IoU=0.5 (greedy match)
  • Per-class AP (from torchmetrics), Per-class ROC-AUC (OVR approx)
  • PR/ROC curves per class (PNG), CSVs, metrics.json
  • Inference timing on test set: time per 100 images + FPS

Usage:
  python evaluate_best_yolo.py \
    --weights /path/to/best.pth \
    --num_classes 13 \
    --val_images /path/to/test/images \
    --val_labels /path/to/test/labels \
    --out_dir eval_out_frcnn_test

Notes
-----
• Faster R-CNN 的 num_classes 如果你“重建头”，需要包含背景类，通常是 N+1。
  如果你的 best.pth 已带正确的头，传 --skip_head_rebuild 以避免重建。
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from torchvision.transforms import functional as F

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay

import matplotlib.pyplot as plt


# ---------------------------
# Dataset (YOLO .txt)
# ---------------------------
class YoloTxtDet(torch.utils.data.Dataset):
    """YOLO txt labels: one .txt per image with lines 'cls cx cy w h' (relative)."""
    def __init__(self, img_dir: str, lbl_dir: str):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.img_paths = sorted([p for p in self.img_dir.iterdir()
                                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        from PIL import Image
        ip = self.img_paths[idx]
        img = Image.open(ip).convert("RGB")
        w, h = img.size

        boxes, labels = [], []
        tp = self.lbl_dir / (ip.stem + ".txt")
        if tp.exists():
            with open(tp, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    c, cx, cy, bw, bh = parts
                    c = int(float(c)) + 1  # shift by +1 (reserve background=0)
                    cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
                    x1 = (cx - bw/2) * w
                    y1 = (cy - bh/2) * h
                    x2 = (cx + bw/2) * w
                    y2 = (cy + bh/2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(c)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return F.to_tensor(img), target


def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)


# ---------------------------
# Model helpers
# ---------------------------
def build_model(num_classes: int, skip_head_rebuild: bool = False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    if not skip_head_rebuild:
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = \
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feat, num_classes)
    return model


def load_weights(model: torch.nn.Module, weights_path: str, device: torch.device):
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    return model


# ---------------------------
# Matching & micro metrics
# ---------------------------
def greedy_match(gt_boxes: torch.Tensor, gt_labels: torch.Tensor,
                 pr_boxes: torch.Tensor, pr_labels: torch.Tensor, pr_scores: torch.Tensor,
                 iou_thresh: float = 0.5) -> Tuple[int, int, int, List[Tuple[int, int, float]]]:
    """Greedy 1-1 matching by score. Return TP, FP, FN, and match list."""
    if pr_boxes.numel():
        order = torch.argsort(pr_scores, descending=True)
        pr_boxes, pr_labels, pr_scores = pr_boxes[order], pr_labels[order], pr_scores[order]

    tp = fp = 0
    fn = int(gt_boxes.size(0))
    used_gt = set()
    matches = []

    if gt_boxes.numel() and pr_boxes.numel():
        ious = box_iou(gt_boxes, pr_boxes)  # [G,P]
        for j in range(pr_boxes.size(0)):
            best_iou, best_gt = torch.max(ious[:, j], dim=0)
            if best_iou >= iou_thresh and (best_gt.item() not in used_gt) and (gt_labels[best_gt] == pr_labels[j]):
                tp += 1
                fn -= 1
                used_gt.add(best_gt.item())
                matches.append((best_gt.item(), j, float(pr_scores[j].item())))
            else:
                fp += 1
    else:
        fp += int(pr_boxes.size(0))

    return tp, fp, fn, matches


def micro_prf1_accuracy(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    acc = tp / (tp + fp + fn + 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}


# ---------------------------
# Evaluation (YOLO only)
# ---------------------------
def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Dataset & loader
    assert args.val_images and args.val_labels, \
        "Provide --val_images and --val_labels (YOLO format)."
    dataset = YoloTxtDet(args.val_images, args.val_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    # Model
    model = build_model(args.num_classes, skip_head_rebuild=args.skip_head_rebuild)
    model = load_weights(model, args.weights, device)
    model.to(device).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Torchmetrics mAP/mAR (per-class on)
    map_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    # Accumulators
    micro_tp = micro_fp = micro_fn = 0
    per_cls_scores: Dict[int, List[float]] = {}
    per_cls_labels: Dict[int, List[int]] = {}

    # ---------- Inference timing ----------
    warmup_batches = 2 if device.type == "cuda" else 0
    seen_images = 0
    t_start = time.perf_counter()

    with torch.no_grad():
        for b_idx, (imgs, targets) in enumerate(loader):
            if b_idx == warmup_batches:
                t_start = time.perf_counter()
                seen_images = 0

            imgs = [img.to(device) for img in imgs]
            preds = model(imgs)

            # score filter
            for i in range(len(preds)):
                keep = preds[i]["scores"].detach().cpu() >= args.score_thresh
                for k in ("boxes", "labels", "scores"):
                    preds[i][k] = preds[i][k][keep]

            # torchmetrics expects cpu tensors
            tm_preds, tm_tgts = [], []
            for i, t in enumerate(targets):
                p = preds[i]
                tm_preds.append({
                    "boxes": p["boxes"].detach().cpu(),
                    "scores": p["scores"].detach().cpu(),
                    "labels": p["labels"].detach().cpu(),
                })
                tm_tgts.append({
                    "boxes": t["boxes"].detach().cpu(),
                    "labels": t["labels"].detach().cpu(),
                })

            map_metric.update(tm_preds, tm_tgts)

            # micro P/R/F1/Acc & AUC prep via greedy match (IoU=0.5)
            for i, t in enumerate(tm_tgts):
                gt_b, gt_l = t["boxes"], t["labels"]
                p = tm_preds[i]
                pr_b, pr_l, pr_s = p["boxes"], p["labels"], p["scores"]

                tp, fp, fn, matches = greedy_match(gt_b, gt_l, pr_b, pr_l, pr_s, iou_thresh=0.5)
                micro_tp += tp; micro_fp += fp; micro_fn += fn

                matched_pred = {pred_idx: 1 for (_, pred_idx, _) in matches}
                for j in range(pr_l.numel()):
                    c = int(pr_l[j].item())
                    per_cls_scores.setdefault(c, []).append(float(pr_s[j].item()))
                    per_cls_labels.setdefault(c, []).append(int(matched_pred.get(j, 0)))

            seen_images += len(imgs)

    # timing
    elapsed = time.perf_counter() - t_start
    avg_per_img = elapsed / max(seen_images, 1)
    per100 = avg_per_img * 100.0
    fps = 1.0 / avg_per_img

    # Compute metrics
    map_res = map_metric.compute()
    overall = {
        "map": float(map_res["map"].item()),
        "map_50": float(map_res["map_50"].item()),
        "map_75": float(map_res["map_75"].item()),
        "mar_100": float(map_res.get("mar_100", torch.tensor(float('nan'))).item()),
    }
    overall.update(micro_prf1_accuracy(micro_tp, micro_fp, micro_fn))

    # Per-class AP (robust extraction)
    def _to_list(x):
        if x is None: return None
        if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
        if isinstance(x, (list, tuple)): return list(x)
        return None

    classes = _to_list(map_res.get("classes", map_res.get("class_indices", None)))
    ap_vals = _to_list(map_res.get("map_per_class", None))
    per_class_ap = {}
    if isinstance(classes, list) and isinstance(ap_vals, list) and len(classes) == len(ap_vals):
        for c, ap in zip(classes, ap_vals):
            try:
                per_class_ap[int(c)] = float(ap)
            except Exception:
                pass

    # Per-class AUC
    per_class_auc = {}
    for c, scores in per_cls_scores.items():
        y_score = np.array(scores)
        y_true = np.array(per_cls_labels[c])
        if len(y_true) >= 2 and (y_true.max() != y_true.min()):
            try:
                auc = roc_auc_score(y_true, y_score)
            except Exception:
                auc = np.nan
        else:
            auc = np.nan
        per_class_auc[c] = float(auc)

    valid_aucs = [v for v in per_class_auc.values() if not np.isnan(v)]
    overall["auc_macro"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    # Timing to overall
    overall["test_time_per_100_imgs_sec"] = float(per100)
    overall["fps"] = float(fps)

    # Save JSON + CSVs
    (out_dir / "metrics.json").write_text(json.dumps({"overall": overall}, indent=2))
    pd.DataFrame([overall]).to_csv(out_dir / "overall_metrics.csv", index=False)

    # Minimal timing CSV for report table
    pd.DataFrame([{
        "model": "Faster R-CNN",
        "dataset": "test",
        "test_time_per_100_imgs_sec": per100,
        "fps": fps
    }]).to_csv(out_dir / "timing.csv", index=False)

    # Per-class CSV (AP + AUC)
    all_classes = sorted(set(list(per_class_ap.keys()) + list(per_class_auc.keys())))
    rows = [{"class_id": c, "AP": per_class_ap.get(c, np.nan), "AUC": per_class_auc.get(c, np.nan)}
            for c in all_classes]
    pd.DataFrame(rows).to_csv(out_dir / "per_class_metrics.csv", index=False)

    # PR/ROC curves
    roc_dir = out_dir / "roc_curves"; roc_dir.mkdir(exist_ok=True)
    pr_dir = out_dir / "pr_curves";  pr_dir.mkdir(exist_ok=True)
    for c in all_classes:
        y_score = np.array(per_cls_scores.get(c, []))
        y_true = np.array(per_cls_labels.get(c, []))
        if len(y_true) >= 2 and (y_true.max() != y_true.min()):
            try:
                RocCurveDisplay.from_predictions(y_true, y_score)
                plt.title(f"ROC (class {c})"); plt.savefig(roc_dir / f"roc_class_{c}.png", bbox_inches="tight"); plt.close()
            except Exception:
                pass
            try:
                PrecisionRecallDisplay.from_predictions(y_true, y_score)
                plt.title(f"PR (class {c})");  plt.savefig(pr_dir / f"pr_class_{c}.png",  bbox_inches="tight"); plt.close()
            except Exception:
                pass

    # Console pretty print
    print("\n[Detection: mAP/mAR]")
    print({k: round(v, 4) for k, v in overall.items() if k in ("map","map_50","map_75","mar_100")})
    print("[P/R/F1/Acc @IoU=0.5]")
    print({k: round(overall[k], 4) for k in ("precision","recall","f1","accuracy")})
    print(f"Macro AUC: {overall['auc_macro']:.4f}")
    print(f"[Timing] per 100 images: {per100:.2f}s | FPS: {fps:.2f}")
    print(f"Saved outputs to: {str(out_dir.resolve())}")


# ---------------------------
# Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Faster R-CNN best.pth on YOLO-format test set")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--num_classes", type=int, required=True,
                   help="If rebuilding head, include background (N+1).")
    p.add_argument("--skip_head_rebuild", action="store_true",
                   help="Use checkpoint head as-is.")
    p.add_argument("--val_images", type=str, required=True,
                   help="YOLO images dir (test set).")
    p.add_argument("--val_labels", type=str, required=True,
                   help="YOLO labels dir (test set).")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--score_thresh", type=float, default=0.05,
                   help="Confidence threshold before evaluation.")
    p.add_argument("--out_dir", type=str, default="eval_out_frcnn_test")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
