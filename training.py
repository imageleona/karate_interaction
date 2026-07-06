"""
training.py -- train the two-person ST-GCN.

Implements section 5 of ``YOLO17_WORKSPACE_SPEC.md``. Reads ``*_train.json`` under ``--data-dir``,
splits off a validation set, trains, and writes everything ``test.py`` needs into a fresh
``output/<timestamp>_training/`` folder: ``config.json``, ``epoch_log.csv``, ``best_model.pth``
(best val accuracy), ``loss_graph.png``, ``accuracy_graph.png``.

Class labels come from ``class_names.json`` in ``--data-dir`` (see dataset.load_class_names), so
the same command trains the 4-class or 8-class set just by pointing at a different tree, e.g.:

  python training.py --data-dir skeleton_dataset_augmented/technique_4class
  python training.py --data-dir skeleton_dataset_augmented/technique_point_8class
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import DEFAULT_NUM_FRAMES, NUM_NODES, SkeletonDataset, load_class_names
from model import STGCN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import wandb
except ImportError:
    wandb = None


def init_wandb(args, run_dir, config):
    """Start a W&B run unless disabled/unavailable. Returns the run or None (never fatal)."""
    if args.no_wandb:
        return None
    if wandb is None:
        print("wandb not installed; skipping W&B logging (pip install wandb).")
        return None
    try:
        return wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=os.path.basename(run_dir),
            mode=args.wandb_mode,
            config=config,
            dir=args.output_root,
            # console capture ("wrapping output streams") intermittently kills the process
            # silently (exit 0) on Windows during rapid successive runs -- disable it.
            # Metrics, config, summaries and images still log normally.
            settings=wandb.Settings(console="off"),
        )
    except Exception as e:  # not logged in, offline, network, etc. -- keep training going
        print(f"wandb.init failed ({e}); continuing without W&B. "
              f"Run 'wandb login' once, or pass --wandb-mode offline / --no-wandb.")
        return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train two-person ST-GCN on COCO-17 skeleton clips.")
    ap.add_argument("--data-dir", default="skeleton_dataset_augmented/technique_4class",
                    help="Root containing <class>_train.json + class_names.json")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-frames", type=int, default=None, help=f"pad/resample T (default {DEFAULT_NUM_FRAMES})")
    ap.add_argument("--interaction-mode", choices=["none", "full", "hand_cross"], default=None,
                    help="cross-person edges (default full; overrides --no-interaction)")
    ap.add_argument("--num-layers", type=int, choices=[4, 6, 9], default=9,
                    help="ST-GCN depth; 6/4 reduce capacity for small datasets (default 9)")
    ap.add_argument("--dropout", type=float, default=0.5,
                    help="dropout inside each ST-GCN block (default 0.5; try 0.6-0.7 vs overfitting)")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="CrossEntropy label smoothing (e.g. 0.1); counters val-loss blowup from overconfidence")
    ap.add_argument("--center", choices=["person", "scene"], default=None,
                    help="hip-centre the coordinates: 'person' = each skeleton on its own hips "
                         "(own-body-relative, best encodes kick height, discards inter-person "
                         "geometry); 'scene' = both on the common hip midpoint (keeps relative "
                         "position). Default: off (raw pixels)")
    ap.add_argument("--scale", action="store_true",
                    help="after centering, divide by the max |coordinate| of the sequence "
                         "(removes camera-zoom / body-size scale)")
    ap.add_argument("--no-interaction", action="store_true", help='shorthand for --interaction-mode none')
    ap.add_argument("--pretrained", default=None,
                    help="Path to an NTU ST-GCN training run dir (or .pth) to initialize the "
                         "backbone from; node-dependent tensors (data_bn, edge_importance, fc) "
                         "stay fresh. E.g. ../nturgb_interaction/output/20260626_012627_del03_interaction")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--tag", default="", help="label appended to the output run folder name")
    ap.add_argument("--no-wandb", action="store_true", help="disable Weights & Biases logging (on by default)")
    ap.add_argument("--wandb-project", default="karate-stgcn", help="wandb project name")
    ap.add_argument("--wandb-entity", default=None, help="wandb entity/team (default: your default)")
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online",
                    help="wandb mode; use 'offline' to log locally without a login")
    return ap.parse_args()


def load_pretrained_backbone(model, path):
    """Transfer shape-compatible weights from an NTU-25 (50-node) ST-GCN checkpoint.

    The conv/BN backbone of the ST-GCN is independent of the node count, so it transfers
    from NTU (num_nodes=50) to COCO-17 (num_nodes=34) directly. Node-dependent tensors are
    kept freshly initialized: ``data_bn`` (C*V), ``gcn.A`` / ``gcn.edge_importance``
    ((3,V,V)), and the class head ``fc``. The NTU reference model names its blocks
    ``layer1..layer9`` while ours is a ModuleList (``layers.0..8``) — keys are remapped.
    Returns (n_transferred_tensors, n_skipped_tensors, transferred_param_count).
    """
    p = path
    if os.path.isdir(p):
        p = os.path.join(p, "best_model.pth")
    state = torch.load(p, map_location="cpu", weights_only=True)

    # group source keys by block ("layer1..layer9" in the NTU reference, "layers.N" here)
    src_blocks: dict[int, dict[str, torch.Tensor]] = {}
    other: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        m = re.match(r"^layer([1-9])\.(.*)$", k) or re.match(r"^layers\.(\d+)\.(.*)$", k)
        if m:
            src_blocks.setdefault(int(m.group(1)), {})[m.group(2)] = v
        else:
            other[k] = v

    # match source blocks to target blocks IN ORDER by gcn-conv shape, so reduced-depth
    # models (num_layers 4/6) still pick up the NTU blocks with the same channel signature
    # (e.g. our 4-layer [2-64, 64-128, 128-256, 256-256] maps to NTU layers 1, 4, 7, 8)
    remapped = dict(other)
    src_ids = sorted(src_blocks)
    si = 0
    for ti, blk in enumerate(model.layers):
        tshape = blk.gcn.conv.weight.shape
        for j in range(si, len(src_ids)):
            w = src_blocks[src_ids[j]].get("gcn.conv.weight")
            if w is not None and w.shape == tshape:
                for sub, v in src_blocks[src_ids[j]].items():
                    remapped[f"layers.{ti}.{sub}"] = v
                si = j + 1
                break

    own = model.state_dict()
    load, skipped = {}, []
    for k, v in remapped.items():
        if k in own and own[k].shape == v.shape:
            load[k] = v
        else:
            skipped.append(k)
    model.load_state_dict(load, strict=False)
    n_params = sum(v.numel() for v in load.values())
    return len(load), len(skipped), n_params


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    torch.set_grad_enabled(True)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    num_frames = args.num_frames if args.num_frames is not None else DEFAULT_NUM_FRAMES
    if args.interaction_mode is not None:
        interaction_mode = args.interaction_mode
    else:
        interaction_mode = "none" if args.no_interaction else "full"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = load_class_names(args.data_dir)
    print(f"data-dir: {args.data_dir}")
    print(f"classes ({len(classes)}): {classes}")
    print(f"device: {device} | interaction_mode: {interaction_mode} | num_frames: {num_frames}")

    norm_kwargs = {"do_center": args.center or False, "do_scale": args.scale}
    train_set = SkeletonDataset(args.data_dir, class_names=classes, mode="train",
                                num_frames=num_frames, **norm_kwargs)
    in_ch = train_set.in_channels
    has_val_files = any(
        os.path.basename(p).endswith("val.json")
        for p in glob.glob(os.path.join(args.data_dir, "**", "*.json"), recursive=True)
    )
    if has_val_files:
        # Preferred: clean, un-augmented val split produced by augment_skeleton_train.py.
        # The split is done on ORIGINAL clips before augmentation, so no augmented copy of a
        # val clip can appear in train -> no data leakage.
        val_set = SkeletonDataset(args.data_dir, class_names=classes, mode="val",
                                  num_frames=num_frames, **norm_kwargs)
        n_train, n_val = len(train_set), len(val_set)
        print(f"samples: train {n_train} (augmented) / val {n_val} (clean, from *_val.json) | in_channels {in_ch}")
    else:
        # Fallback: no val files. Random split of the train set. NOTE: if this data dir was
        # augmented, this reintroduces leakage (aug copies of a clip split across train/val).
        # Regenerate with augment_skeleton_train.py (writes *_val.json) to avoid this.
        n_val = max(1, int(len(train_set) * args.val_ratio))
        n_train = len(train_set) - n_val
        gen = torch.Generator().manual_seed(args.seed)
        train_set, val_set = random_split(train_set, [n_train, n_val], generator=gen)
        print(f"WARNING: no *_val.json found; using random_split of train (val {n_val}). "
              f"If this dir is augmented, val is LEAKY -- regenerate to get clean *_val.json.")
        print(f"samples: train {n_train} / val {n_val} | in_channels {in_ch}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = STGCN(num_classes=len(classes), in_channels=in_ch,
                  num_nodes=NUM_NODES, interaction_mode=interaction_mode,
                  dropout=args.dropout, num_layers=args.num_layers).to(device)
    if args.pretrained:
        n_loaded, n_skipped, n_params = load_pretrained_backbone(model, args.pretrained)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"pretrained: {args.pretrained}")
        print(f"  transferred {n_loaded} tensors ({n_params:,} params = "
              f"{n_params / total_params:.0%} of model), skipped {n_skipped} "
              f"(node-dependent: data_bn/A/edge_importance/fc + NTU residuals)")
        model = model.to(device)
    # Exclude edge_importance from weight decay: decay shrinks all its entries toward 0
    # uniformly (~6%/40ep), drowning the small data-driven differences we actually want
    # the graph to learn (and want to visualize).
    decay_params, no_decay_params = [], []
    for pname, p in model.named_parameters():
        (no_decay_params if "edge_importance" in pname else decay_params).append(p)
    optimizer = torch.optim.Adam(
        [{"params": decay_params, "weight_decay": 5e-4},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = os.path.join(args.output_root, f"{ts}{tag}_training")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"in_channels": in_ch, "num_nodes": NUM_NODES, "interaction_mode": interaction_mode,
                   "classes": classes, "num_frames": num_frames,
                   "num_layers": args.num_layers, "dropout": args.dropout,
                   "do_center": args.center or "", "do_scale": bool(args.scale),
                   "pretrained": args.pretrained or ""}, f, indent=2)

    run = init_wandb(args, run_dir, {
        "pretrained": args.pretrained or "",
        "num_layers": args.num_layers, "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "do_center": args.center or "", "do_scale": bool(args.scale),
        "in_channels": in_ch, "num_nodes": NUM_NODES, "interaction_mode": interaction_mode,
        "num_classes": len(classes), "classes": classes, "num_frames": num_frames,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "weight_decay": 5e-4,
        "val_ratio": args.val_ratio, "seed": args.seed, "optimizer": "adam",
        "scheduler": "multistep", "lr_milestones": milestones, "data_dir": args.data_dir,
        "dataset_tag": args.tag, "n_train": n_train, "n_val": n_val, "device": str(device),
    })

    log_path = os.path.join(run_dir, "epoch_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        for k, v in [("train_loss", tr_loss), ("val_loss", va_loss),
                     ("train_acc", tr_acc), ("val_acc", va_acc)]:
            history[k].append(v)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, lr_now, f"{tr_loss:.6f}", f"{tr_acc:.6f}",
                                    f"{va_loss:.6f}", f"{va_acc:.6f}"])
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
        if run is not None:
            run.log({"epoch": epoch, "lr": lr_now,
                     "train/loss": tr_loss, "train/acc": tr_acc,
                     "val/loss": va_loss, "val/acc": va_acc, "val/best_acc": best_val}, step=epoch)
        print(f"epoch {epoch:3d}/{args.epochs}  lr {lr_now:.2e}  "
              f"train {tr_loss:.4f}/{tr_acc:.3f}  val {va_loss:.4f}/{va_acc:.3f}  best {best_val:.3f}")

    # if val never improved above the -1 sentinel (shouldn't happen), still save a checkpoint
    if not os.path.exists(os.path.join(run_dir, "best_model.pth")):
        torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))

    _plot(history["train_loss"], history["val_loss"], "Loss",
          os.path.join(run_dir, "loss_graph.png"))
    _plot(history["train_acc"], history["val_acc"], "Accuracy",
          os.path.join(run_dir, "accuracy_graph.png"))

    if run is not None:
        run.summary["best_val_acc"] = best_val
        for name in ("loss_graph.png", "accuracy_graph.png"):
            p = os.path.join(run_dir, name)
            if os.path.exists(p):
                run.log({name[:-4]: wandb.Image(p)})
        run.finish()

    # record the completed run dir so a following step (e.g. test.py in a batch) can target it exactly
    with open(os.path.join(args.output_root, "latest_run.txt"), "w", encoding="utf-8") as f:
        f.write(run_dir + "\n")

    print(f"\nDone. best val acc = {best_val:.4f}")
    print(f"run dir: {run_dir}")


def _plot(train_vals, val_vals, ylabel, path) -> None:
    plt.figure()
    plt.plot(range(1, len(train_vals) + 1), train_vals, label=f"train {ylabel.lower()}")
    plt.plot(range(1, len(val_vals) + 1), val_vals, label=f"val {ylabel.lower()}")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
