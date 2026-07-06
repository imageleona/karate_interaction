"""Paper-ready figure: layer-averaged learned cross-person edge multipliers,
4class vs 8class side by side.

Saves output/edge_importance_paper.{png,svg,pdf} -- no in-image title (the caption
belongs in the paper); .svg/.pdf are vector files editable in Inkscape/Illustrator.
Edit RUNS below to point at different checkpoints."""
import json, os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
from model import STGCN

NAMES = ["nose", "L eye", "R eye", "L ear", "R ear", "L shoulder", "R shoulder",
         "L elbow", "R elbow", "L wrist", "R wrist", "L hip", "R hip",
         "L knee", "R knee", "L ankle", "R ankle"]

RUNS = [
    ("(a) 4-class: technique", "output/20260705_161105_4class_full_pt_norm_s43_training"),
    ("(b) 8-class: technique x point", "output/20260705_163924_8class_full_pt_norm_s42_training"),
]

def mean_cross(run):
    cfg = json.load(open(os.path.join(run, "config.json")))
    m = STGCN(num_classes=len(cfg["classes"]), in_channels=cfg["in_channels"],
              num_nodes=cfg["num_nodes"], interaction_mode=cfg["interaction_mode"],
              num_layers=cfg.get("num_layers", 9))
    m.load_state_dict(torch.load(os.path.join(run, "best_model.pth"),
                                 map_location="cpu", weights_only=True))
    mults = []
    for blk in m.layers:
        A = blk.gcn.A.detach().numpy()
        imp = blk.gcn.edge_importance.detach().numpy()
        a_sum = A.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mm = (A * imp).sum(0) / a_sum
        mm[a_sum == 0] = np.nan
        mults.append(mm)
    mean_m = np.nanmean(np.stack(mults), axis=0)
    return (mean_m[:17, 17:] + mean_m[17:, :17].T) / 2      # (17,17) symmetrized

crosses = [(title, mean_cross(run)) for title, run in RUNS]
span = max(np.nanmax(np.abs(c - 1.0)) for _, c in crosses)
norm = TwoSlopeNorm(vcenter=1.0, vmin=1.0 - span, vmax=1.0 + span)
cmap = plt.get_cmap("coolwarm")

fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0))
for k, (ax, (title, cross)) in enumerate(zip(axes, crosses)):
    im = ax.imshow(cross, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Person 2 joint", fontsize=11)
    if k == 0:
        ax.set_ylabel("Person 1 joint", fontsize=11)
    ax.set_xticks(range(17))
    ax.set_xticklabels(NAMES, rotation=90, fontsize=8.5)
    ax.set_yticks(range(17))
    ax.set_yticklabels(NAMES, fontsize=8.5)
    # subtle grid for readability
    ax.set_xticks(np.arange(-0.5, 17), minor=True)
    ax.set_yticks(np.arange(-0.5, 17), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", length=0)

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("learned edge multiplier (1.0 = unchanged by training)", fontsize=11)
for ext, kw in [("png", {"dpi": 200}), ("svg", {}), ("pdf", {})]:
    out = f"output/edge_importance_paper.{ext}"
    fig.savefig(out, bbox_inches="tight", **kw)
    print("saved", out)
plt.close(fig)
