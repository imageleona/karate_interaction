"""
test.py -- evaluate a trained ST-GCN on the test split.

Implements section 6 of ``YOLO17_WORKSPACE_SPEC.md``. Loads a training run's ``config.json`` +
``best_model.pth`` (auto-picking the newest ``output/*_training/`` if ``--output-dir`` is
omitted), rebuilds the model straight from config, runs inference over ``*_test.json`` under
``--data-dir``, and writes a fresh ``output/<timestamp>_test/`` with:
  * ``test_report.txt``     -- sklearn classification_report (per-class precision/recall/F1)
  * ``test_cm.png`` / ``test_cm.txt`` -- row-normalized confusion matrix (heatmap + text)
  * ``checkpoint_source.txt`` -- the training dir used

Point ``--data-dir`` at the ORIGINAL (un-augmented) tree for an honest test set, e.g.:

  python test.py --data-dir skeleton_dataset/technique_4class
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import NUM_NODES, SkeletonDataset
from model import STGCN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate a trained ST-GCN on the test split.")
    ap.add_argument("--output-dir", default=None,
                    help="A ..._training run folder. If omitted, newest valid output/*_training/ is used.")
    ap.add_argument("--data-dir", default="skeleton_dataset/technique_4class",
                    help="Root containing <class>_test.json + class_names.json")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--device", default="auto",
                    help="'auto' (cuda if available), 'cpu', 'cuda', ...")
    return ap.parse_args()


def find_latest_training(output_root: str) -> str:
    runs = sorted(glob.glob(os.path.join(output_root, "*_training")))
    for run in reversed(runs):
        if (os.path.exists(os.path.join(run, "config.json"))
                and os.path.exists(os.path.join(run, "best_model.pth"))):
            return run
    raise SystemExit(f"No valid *_training run (config.json + best_model.pth) under {output_root}")


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_dir = args.output_dir or find_latest_training(args.output_root)
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    classes = cfg["classes"]
    print(f"checkpoint: {run_dir}")
    print(f"classes ({len(classes)}): {classes}")
    print(f"device: {device}")

    camera_csv = cfg.get("camera_angle_csv") or None
    if camera_csv and not os.path.exists(camera_csv):
        raise SystemExit(f"config.json references camera_angle_csv={camera_csv} but the file "
                         f"is missing; the model needs the camera feature at test time")
    ds = SkeletonDataset(args.data_dir, class_names=classes, mode="test",
                         num_frames=cfg["num_frames"],
                         do_center=cfg.get("do_center") or False,
                         do_scale=cfg.get("do_scale", False),
                         camera_angle_csv=camera_csv)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"test samples: {len(ds)}")
    if camera_csv:
        print(f"camera-angle feature: {camera_csv}")

    model = STGCN(num_classes=len(classes), in_channels=cfg["in_channels"],
                  num_nodes=cfg.get("num_nodes", NUM_NODES),
                  interaction_mode=cfg["interaction_mode"],
                  num_layers=cfg.get("num_layers", 9),
                  extra_feature_dim=cfg.get("extra_feature_dim", 0)).to(device)
    state = torch.load(os.path.join(run_dir, "best_model.pth"), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            extra = batch[2].to(device) if len(batch) > 2 else None
            logits = model(batch[0].to(device), extra)
            all_pred.append(logits.argmax(1).cpu().numpy())
            all_true.append(batch[1].numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)

    labels = list(range(len(classes)))
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=classes, digits=4, zero_division=0)
    acc = float((y_pred == y_true).mean())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_norm / row_sums

    ts = time.strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join(args.output_root, f"{ts}_test")
    os.makedirs(test_dir, exist_ok=True)

    with open(os.path.join(test_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {run_dir}\n")
        f.write(f"data-dir:   {args.data_dir}\n")
        f.write(f"test samples: {len(ds)}\n")
        f.write(f"overall accuracy: {acc:.4f}\n\n")
        f.write(report + "\n")

    with open(os.path.join(test_dir, "test_cm.txt"), "w", encoding="utf-8") as f:
        f.write("row-normalized confusion matrix (rows=true, cols=pred)\n")
        f.write("classes: " + ", ".join(classes) + "\n\n")
        for i, row in enumerate(cm_norm):
            f.write(f"{classes[i]:>20s}  " + "  ".join(f"{v:.3f}" for v in row) + "\n")

    _plot_cm(cm_norm, classes, os.path.join(test_dir, "test_cm.png"))

    # per-clip predictions (needs the clip_ids embedded in the dataset JSONs) -- the
    # join key for downstream analyses such as accuracy-vs-camera-angle
    with open(os.path.join(test_dir, "per_clip_predictions.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "true", "pred", "true_class", "pred_class", "correct"])
        for cid, t, p in zip(ds.clip_ids, y_true, y_pred):
            w.writerow([cid or "", int(t), int(p), classes[t], classes[p], int(t == p)])

    with open(os.path.join(test_dir, "checkpoint_source.txt"), "w", encoding="utf-8") as f:
        f.write(run_dir + "\n")

    print(f"\noverall accuracy: {acc:.4f}")
    print(report)
    print(f"results: {test_dir}")


def _plot_cm(cm_norm: np.ndarray, classes: list[str], path: str) -> None:
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(4, n * 0.8), max(4, n * 0.8)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
