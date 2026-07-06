#!/usr/bin/env python
# coding=utf-8

"""
Visualize learned edge-importance weights from a trained two-person COCO-17 ST-GCN.

Adapted from nturgb_interaction/visualize_edges.py for this workspace's model
(COCO-17, 34 nodes, layers held in an nn.ModuleList). For each of the 9 ST-GCN layers:
  - Left panel  : two skeletons with bones colored by the LEARNED within-person multiplier
  - Right panel : 17x17 heatmap of the LEARNED cross-person multiplier (P1 x P2 joints)

Shown value = A-weighted mean of ``edge_importance`` per edge, which initializes to exactly
1.0 -- so red (>1) / blue (<1) is purely what training changed. The raw ``(A*imp).sum(0)``
plotted previously is dominated by the adjacency's row normalization (the BFS-center left-hip
rows are ~8x larger BY CONSTRUCTION, before any training), which made every run look like
"only the left hips connect" regardless of learning.

Usage:
    python visualize_edges.py                                   # latest output/*_training
    python visualize_edges.py --output-dir output/<ts>_training
    python visualize_edges.py --output-dir output/<ts>_training --save-path fig.png
"""

import argparse
import json
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from model import COCO17_EDGES, STGCN

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_ROOT = os.path.join(_SCRIPT_DIR, "output")

JOINTS_PER_PERSON = 17

# 2D canonical positions for COCO-17 joints (x right, y up) -- a facing stick figure.
JOINT_POS = np.array([
    [ 0.00,  1.00],  #  0 nose
    [-0.06,  1.07],  #  1 left eye
    [ 0.06,  1.07],  #  2 right eye
    [-0.13,  1.03],  #  3 left ear
    [ 0.13,  1.03],  #  4 right ear
    [-0.28,  0.75],  #  5 left shoulder
    [ 0.28,  0.75],  #  6 right shoulder
    [-0.42,  0.45],  #  7 left elbow
    [ 0.42,  0.45],  #  8 right elbow
    [-0.50,  0.15],  #  9 left wrist
    [ 0.50,  0.15],  # 10 right wrist
    [-0.18,  0.25],  # 11 left hip
    [ 0.18,  0.25],  # 12 right hip
    [-0.20, -0.15],  # 13 left knee
    [ 0.20, -0.15],  # 14 right knee
    [-0.22, -0.55],  # 15 left ankle
    [ 0.22, -0.55],  # 16 right ankle
])

JOINT_NAMES = [
    "nose", "L eye", "R eye", "L ear", "R ear",
    "L shldr", "R shldr", "L elbow", "R elbow",
    "L wrist", "R wrist", "L hip", "R hip",
    "L knee", "R knee", "L ankle", "R ankle",
]


def _find_latest_training_dir() -> Optional[str]:
    if not os.path.isdir(_OUTPUT_ROOT):
        return None
    best_path, best_mtime = None, -1.0
    for name in os.listdir(_OUTPUT_ROOT):
        if not name.endswith("_training"):
            continue
        path = os.path.join(_OUTPUT_ROOT, name)
        if not (os.path.isfile(os.path.join(path, "config.json")) and
                os.path.isfile(os.path.join(path, "best_model.pth"))):
            continue
        m = os.path.getmtime(path)
        if m > best_mtime:
            best_mtime, best_path = m, path
    return best_path


def _get_layer_learned_multiplier(gcn_module) -> np.ndarray:
    """Learned edge multiplier per (i, j): A-weighted mean of edge_importance -> (V, V).

    edge_importance initializes to 1 everywhere, so this matrix is exactly 1.0 before any
    training and deviations from 1 are what the model LEARNED. (The raw ``(A*imp).sum(0)``
    used previously is dominated by the adjacency's row normalization -- e.g. the left-hip
    BFS-center rows are ~8x larger by construction -- which hides the learning entirely.)
    Cells with no edge are NaN.
    """
    A = gcn_module.A.detach().cpu().numpy()                  # (3, V, V)
    imp = gcn_module.edge_importance.detach().cpu().numpy()  # (3, V, V)
    a_sum = A.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mult = (A * imp).sum(axis=0) / a_sum
    mult[a_sum == 0] = np.nan
    return mult                                              # (V, V), init == 1.0


def _diverging_norm(values, min_span: float = 0.01) -> Normalize:
    """Symmetric normalization around 1.0 (the init value) for a diverging colormap."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    d = float(np.max(np.abs(vals - 1.0))) if vals.size else 0.0
    d = max(d, min_span)
    return Normalize(vmin=1.0 - d, vmax=1.0 + d)


def _draw_skeleton(ax, joint_pos, edge_weights, cmap, norm, person_label, label_x=0.5):
    span = max(norm.vmax - 1.0, 1e-9)
    for (i, j), w in edge_weights.items():
        x = [joint_pos[i, 0], joint_pos[j, 0]]
        y = [joint_pos[i, 1], joint_pos[j, 1]]
        rel = min(abs(w - 1.0) / span, 1.0)       # thickness = |deviation from init|
        ax.plot(x, y, color=cmap(norm(w)), linewidth=1.5 + 4.0 * rel, solid_capstyle="round")
    ax.scatter(joint_pos[:, 0], joint_pos[:, 1], c="white", s=28, zorder=5,
               edgecolors="grey", linewidths=0.5)
    ax.text(label_x, -0.04, person_label, transform=ax.transAxes, ha="center", fontsize=8, color="grey")


def visualize(output_dir: str, save_path: str):
    with open(os.path.join(output_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    interaction_mode = cfg.get("interaction_mode", "full")
    model = STGCN(
        num_classes=len(cfg["classes"]),
        in_channels=cfg["in_channels"],
        num_nodes=cfg["num_nodes"],
        interaction_mode=interaction_mode,
        num_layers=cfg.get("num_layers", 9),
    )
    state = torch.load(os.path.join(output_dir, "best_model.pth"),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    p = JOINTS_PER_PERSON
    layers = [blk.gcn for blk in model.layers]              # ModuleList of STGCNBlocks
    layer_names = []
    for i, blk in enumerate(model.layers):                  # derive names from the blocks
        c_out3, c_in = blk.gcn.conv.weight.shape[:2]
        stride = blk.tcn[2].stride[0]
        layer_names.append(f"Layer {i + 1} ({c_in}→{c_out3 // 3}"
                           + (", stride 2)" if stride == 2 else ")"))

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#2a2a3a")                                # cells with no edge
    p1_offset = np.array([-1.6, 0.0])
    p2_offset = np.array([1.6, 0.0])

    n_rows = len(layers)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 46 * n_rows / 9.0 + 1))
    title = (f"LEARNED edge multiplier per ST-GCN layer — deviation from init 1.0 "
             f"(red > 1 strengthened, blue < 1 weakened)  |  interaction_mode = {interaction_mode}")
    fig.suptitle(title, fontsize=11, y=0.995)

    for row, (gcn, lname) in enumerate(zip(layers, layer_names)):
        mult = _get_layer_learned_multiplier(gcn)          # (34, 34), init == 1.0, NaN = no edge

        # --- within-person bone multipliers (symmetrized) ---
        within_p1, within_p2 = {}, {}
        for (i, j) in COCO17_EDGES:
            within_p1[(i, j)] = (mult[i, j] + mult[j, i]) / 2
            within_p2[(i, j)] = (mult[i + p, j + p] + mult[j + p, i + p]) / 2
        cross = (mult[:p, p:] + mult[p:, :p].T) / 2        # (17, 17)
        # one shared scale per layer so skeleton and heatmap are comparable
        norm = _diverging_norm(list(within_p1.values()) + list(within_p2.values())
                               + list(cross[np.isfinite(cross)].ravel()))

        ax_skel = axes[row, 0]
        ax_skel.set_facecolor("#1a1a2e")
        ax_skel.set_aspect("equal")
        ax_skel.axis("off")
        ax_skel.set_title(lname, fontsize=9, pad=4)
        _draw_skeleton(ax_skel, JOINT_POS + p1_offset, within_p1, cmap, norm, "Person 1", label_x=0.27)
        _draw_skeleton(ax_skel, JOINT_POS + p2_offset, within_p2, cmap, norm, "Person 2", label_x=0.73)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax_skel, fraction=0.02, pad=0.02, label="learned multiplier")

        # --- cross-person heatmap (P1 joints x P2 joints), both directions averaged ---
        ax_heat = axes[row, 1]
        im = ax_heat.imshow(np.ma.masked_invalid(cross), cmap=cmap, norm=norm, aspect="auto")
        ax_heat.set_title(f"{lname} — cross-person (learned multiplier)", fontsize=9, pad=4)
        ax_heat.set_xlabel("Person 2 joint", fontsize=7)
        ax_heat.set_ylabel("Person 1 joint", fontsize=7)
        ax_heat.set_xticks(range(p))
        ax_heat.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=6)
        ax_heat.set_yticks(range(p))
        ax_heat.set_yticklabels(JOINT_NAMES, fontsize=6)
        if not np.isfinite(cross).any():
            ax_heat.text(0.5, 0.5, "no cross-person edges\n(interaction_mode = none)",
                         transform=ax_heat.transAxes, ha="center", va="center",
                         fontsize=9, color="grey")
        fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02, label="learned multiplier")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"interaction_mode: {interaction_mode} | classes: {cfg['classes']}")
    print(f"Saved to {save_path}")


def main():
    ap = argparse.ArgumentParser(description="Visualize learned ST-GCN edge importance.")
    ap.add_argument("--output-dir", default=None,
                    help="Training run folder (default: latest output/*_training)")
    ap.add_argument("--save-path", default=None,
                    help="Where to save the figure (default: <output-dir>/edge_importance.png)")
    args = ap.parse_args()

    output_dir = args.output_dir or _find_latest_training_dir()
    if not output_dir:
        print("No *_training folder found. Run training.py first or pass --output-dir.")
        return
    output_dir = os.path.abspath(output_dir)
    save_path = args.save_path or os.path.join(output_dir, "edge_importance.png")
    visualize(output_dir, save_path)


if __name__ == "__main__":
    main()
