"""Presentation figures: 4-class vs 8-class vs 15-class task comparison.

Uses the 20260730 full_pt_norm runs (no camera feature), seeds 42-44 -- the only
sweep where all three tasks share identical settings. Saves to output/:

  task_comparison_accuracy.tex             LaTeX table of val + TEST accuracy
  task_comparison_accuracy.{png,svg,pdf}   the same numbers as grouped bars
  task_comparison_confusion.{png,svg,pdf}  all three test confusion matrices
  task_confusion_<n>class.{png,svg,pdf}    one matrix per task, for its own slide

Confusion matrices are row-normalized (= per-class recall) and pooled over the
three seeds, so each row sums to 1 and the diagonal is directly comparable. They
are drawn by test.py's own _plot_cm, so a pooled matrix here is styled exactly
like the per-run output/*_test/test_cm.png -- every cell labelled, 0.00 included.
"""
import csv
import glob
import os
import re
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# the confusion-matrix renderer test.py uses for every run, reused verbatim so the
# pooled matrices match output/*_test/test_cm.png exactly
from test import _plot_cm

# --- palette (validated: adjacent CVD dE 24.7, normal-vision 33.6, contrast >=3:1) ---
VAL_C = "#2a78d6"      # categorical slot 1, blue
TEST_C = "#eb6834"     # categorical slot 2, orange
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"

TASKS = [("4class", "4-class\ntechnique"), ("8class", "8-class\ntechnique x point"),
         ("15class", "15-class\ntechnique x point x reason")]
RUN_RE = re.compile(r"^\d{8}_\d{6}_(\d+class)_full_pt_norm_s(\d+)_training$")
SEEDS = {"42", "43", "44"}
STAMP = "20260730"     # the sweep that contains all three tasks


def collect():
    """task -> {'val': [...], 'test': [...], 'cm': (n,n) counts, 'classes': [...]}"""
    test_dir = {}
    for td in sorted(glob.glob("output/*_test")):
        src = os.path.join(td, "checkpoint_source.txt")
        if os.path.exists(src):
            ck = os.path.basename(open(src, encoding="utf-8").read().strip().replace("\\", "/"))
            test_dir[ck] = td      # later dirs win, matching summarize_results.py

    out = {}
    for run in sorted(glob.glob(f"output/{STAMP}_*_training")):
        base = os.path.basename(run)
        m = RUN_RE.match(base)
        if not m or m.group(2) not in SEEDS:
            continue
        task = m.group(1)
        pred = os.path.join(test_dir.get(base, ""), "per_clip_predictions.csv")
        if not os.path.exists(pred):
            print(f"  (skipping {base}: no test predictions)")
            continue
        rows = list(csv.DictReader(open(pred, encoding="utf-8")))
        va = [float(r["val_acc"]) for r in
              csv.DictReader(open(os.path.join(run, "epoch_log.csv"), encoding="utf-8"))]

        d = out.setdefault(task, {"val": [], "test": [], "f1": [], "preds": [],
                                  "cm": None, "classes": None})
        if d["classes"] is None:
            # class order = label index order used at test time
            idx = {}
            for r in rows:
                idx[int(r["true"])] = r["true_class"]
                idx[int(r["pred"])] = r["pred_class"]
            d["classes"] = [idx[i] for i in sorted(idx)]
            d["cm"] = np.zeros((len(d["classes"]), len(d["classes"])), int)
        d["val"].append(max(va))
        d["test"].append(sum(int(r["correct"]) for r in rows) / len(rows))
        d["f1"].append(macro_f1(rows, len(d["classes"])))
        d["preds"].append(rows)
        for r in rows:
            d["cm"][int(r["true"]), int(r["pred"])] += 1
    return out


def macro_f1(rows, n_classes: int) -> float:
    """Unweighted mean per-class F1, matching sklearn's macro avg (zero_division=0)."""
    f1s = []
    for lab in range(n_classes):
        tp = sum(int(r["true"]) == lab and int(r["pred"]) == lab for r in rows)
        fp = sum(int(r["true"]) != lab and int(r["pred"]) == lab for r in rows)
        fn = sum(int(r["true"]) == lab and int(r["pred"]) != lab for r in rows)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * rc / (p + rc) if p + rc else 0.0)
    return sum(f1s) / len(f1s)


def save(fig, stem):
    for ext, kw in [("png", {"dpi": 200}), ("svg", {}), ("pdf", {})]:
        path = f"output/{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
        print("saved", path)
    plt.close(fig)


# --------------------------- table 1: accuracy ---------------------------

def latex_table(data):
    """booktabs table of the same numbers as the accuracy figure.

    Row labels, caption and label are plain text on their own lines so they are
    easy to rewrite; \\pm cells are $...$ so they work in any column type.
    """
    rows = []
    for task, _ in TASKS:
        d = data[task]
        n = len(d["classes"])
        cells = [f"{n}", f"{1.0 / n:.3f}"]
        for key in ("val", "test", "f1"):
            v = np.array(d[key])
            cells.append(f"${v.mean():.3f} \\pm {v.std(ddof=0):.3f}$")
        per = ", ".join(f"{x:.3f}" for x in d["test"])
        rows.append((f"{n}-class", cells, per))

    body = "\n".join("    " + f"{name:<10s} & " + " & ".join(c) + r" \\"
                     for name, c, _ in rows)
    seeds = "; ".join(f"{name}: {per}" for name, _, per in rows)

    tex = rf"""% requires \usepackage{{booktabs}}
% generated by make_task_comparison_figures.py -- edit the labels freely
\begin{{table}}[t]
  \centering
  \caption{{Task granularity comparison. ST-GCN with cross-person interaction
    edges, NTU-pretrained, scene-centred and scale-normalised skeletons.
    Mean $\pm$ standard deviation over seeds 42--44. Validation is the best
    epoch on 106 held-out clips; TEST is 162--163 clips from eight videos held
    out entirely from training.}}
  \label{{tab:task-comparison}}
  \begin{{tabular}}{{lccccc}}
    \toprule
    Task & Classes & Chance & Val.\ acc. & Test acc. & Macro-F1 \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
  \\[2pt]
  \footnotesize Per-seed test accuracy --- {seeds}.
\end{{table}}
"""
    path = "output/task_comparison_accuracy.tex"
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print("saved", path)
    print(tex)


# ------------------- table 2: collapsing to a coarser task -------------------

def to_8(label: str) -> str:
    """15-class label -> its 8-class label: technique + point/no-point."""
    tech = label.split("_")[0]
    return f"{tech}_no_point" if "no_point" in label else f"{tech}_point"


def to_2(label: str) -> str:
    return "no_point" if "no_point" in label else "point"


def collapse_scores(preds, fn):
    """(acc, macro-F1) after mapping both true and pred through fn."""
    true = [fn(r["true_class"]) for r in preds]
    pred = [fn(r["pred_class"]) for r in preds]
    acc = sum(t == p for t, p in zip(true, pred)) / len(true)
    f1s = []
    for lab in sorted(set(true) | set(pred)):
        tp = sum(t == lab and p == lab for t, p in zip(true, pred))
        fp = sum(t != lab and p == lab for t, p in zip(true, pred))
        fn_ = sum(t == lab and p != lab for t, p in zip(true, pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn_) if tp + fn_ else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return acc, sum(f1s) / len(f1s)


def latex_collapse_table(preds_by_task):
    """8-class and 15-class models scored on the coarser tasks they contain."""
    ident = lambda s: s
    rows = [("15-class", {"15class": ident}),
            ("8-class", {"15class": to_8, "8class": ident}),
            ("Point / no-point", {"15class": to_2, "8class": to_2})]

    body = []
    for name, fns in rows:
        cells = []
        for task in ("8class", "15class"):
            if task not in fns:
                cells += [r"---", r"---"]
                continue
            got = [collapse_scores(p, fns[task]) for p in preds_by_task[task]]
            for k in (0, 1):
                v = np.array([g[k] for g in got])
                cells.append(f"${v.mean():.3f} \\pm {v.std(ddof=0):.3f}$")
        body.append(f"    {name:<16s} & " + " & ".join(cells) + r" \\")

    tex = r"""\documentclass[border=4pt]{standalone}

\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{array}

\begin{document}

\small
\renewcommand{\arraystretch}{1.2}
\setlength{\tabcolsep}{7pt}

  \begin{tabular}{lcccc}
    \toprule
    & \multicolumn{2}{c}{8-class model} & \multicolumn{2}{c}{15-class model} \\
    \cmidrule(lr){2-3} \cmidrule(lr){4-5}
    Evaluated as & Acc. & Macro-F1 & Acc. & Macro-F1 \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}

\end{document}
"""
    path = "output/task_collapse_comparison.tex"
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print("saved", path)
    print(tex)


# --------------------------- figure 1: accuracy ---------------------------

def accuracy_figure(data):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = np.arange(len(TASKS))
    w = 0.34
    rng = np.random.default_rng(0)

    for k, (key, color, label) in enumerate([
            ("val", VAL_C, "Validation (best epoch, 106 clips)"),
            ("test", TEST_C, "TEST (held-out videos, ~163 clips)")]):
        vals = [np.array(data[t]["val" if key == "val" else "test"]) for t, _ in TASKS]
        means = [v.mean() for v in vals]
        stds = [v.std(ddof=0) for v in vals]
        pos = x + (k - 0.5) * w
        ax.bar(pos, means, w, color=color, label=label, zorder=3,
               edgecolor="white", linewidth=2)
        ax.errorbar(pos, means, yerr=stds, fmt="none", ecolor=INK_2,
                    elinewidth=1.6, capsize=6, capthick=1.6, zorder=5)
        # per-seed dots: the spread is the story, so show it
        for p, v in zip(pos, vals):
            ax.scatter(p + rng.uniform(-0.055, 0.055, len(v)), v, s=26,
                       facecolor="white", edgecolor=INK, linewidth=1.1, zorder=6)
        # label above the error-bar cap so it never lands on a seed dot
        for p, m, s in zip(pos, means, stds):
            ax.text(p, m + s + 0.025, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=13, color=INK, fontweight="bold", zorder=7)

    # chance level per task
    for xi, (t, _) in zip(x, TASKS):
        c = 1.0 / len(data[t]["classes"])
        ax.plot([xi - 0.5, xi + 0.5], [c, c], ls=(0, (4, 3)), color=INK_2,
                linewidth=1.5, zorder=4)
        ax.text(xi + 0.5, c, f"  chance {c:.2f}", va="center", ha="left",
                fontsize=10.5, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in TASKS], fontsize=14)
    ax.set_ylabel("accuracy", fontsize=15, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="y", labelsize=12, colors=INK_2)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    # inside the axes: the right-hand columns are short, so this space is free
    ax.legend(fontsize=12.5, frameon=False, loc="upper right", ncol=1)
    ax.set_title("Accuracy falls far slower than the label space grows\n"
                 "mean +/- SD over seeds 42-44; dots = individual seeds",
                 fontsize=15, color=INK, loc="left", pad=18)
    save(fig, "task_comparison_accuracy")


# ------------------------ figure 2: confusion matrices ------------------------

def row_normalize(cm):
    """Same normalization as test.py: divide by row sum, guarding empty rows."""
    norm = cm.astype(np.float64)
    row_sums = norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return norm / row_sums


def confusion_figures(data):
    for t, _ in TASKS:
        norm = row_normalize(data[t]["cm"])
        classes = data[t]["classes"]
        # test.py's own renderer -- every cell labelled to 2dp including 0.00.
        # It hardcodes dpi=120, which vector backends ignore, so one call per format.
        for ext in ("png", "svg", "pdf"):
            path = f"output/task_confusion_{t}.{ext}"
            _plot_cm(norm, classes, path)
            print("saved", path)

        # text dump alongside, mirroring output/*_test/test_cm.txt
        path = f"output/task_confusion_{t}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("row-normalized confusion matrix (rows=true, cols=pred)\n")
            f.write(f"pooled over seeds 42-44; mean test acc {np.mean(data[t]['test']):.4f}\n")
            f.write("classes: " + ", ".join(classes) + "\n\n")
            for i, row in enumerate(norm):
                f.write(f"{classes[i]:>28s}  " + "  ".join(f"{v:.3f}" for v in row) + "\n")
        print("saved", path)


if __name__ == "__main__":
    data = collect()
    missing = [t for t, _ in TASKS if t not in data]
    if missing:
        raise SystemExit(f"no runs found for {missing}")
    for t, _ in TASKS:
        print(f"{t}: {len(data[t]['test'])} seeds, {len(data[t]['classes'])} classes, "
              f"val {np.mean(data[t]['val']):.3f} test {np.mean(data[t]['test']):.3f}")
    latex_table(data)
    latex_collapse_table({t: data[t]["preds"] for t, _ in TASKS})
    accuracy_figure(data)
    confusion_figures(data)
