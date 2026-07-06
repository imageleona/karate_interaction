"""
summarize_results.py -- aggregate final-comparison runs into a mean +/- std table.

Scans output/*_training runs whose tag matches
    <dataset>_<interaction>_<init>_s<seed>   e.g.  4class_full_pt_s42
finds each run's matching test dir (via the test dir's checkpoint_source.txt),
and groups the seeds per (dataset, interaction, init) cell.

Writes output/final_comparison_summary.csv and prints a markdown table with
best-val and TEST accuracy as mean +/- std over seeds.

Usage:
    python summarize_results.py                  # scan ./output
    python summarize_results.py --output-root output
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np

TAG_RE = re.compile(
    r"^\d{8}_\d{6}_((4|8)class)_(full|none|hand_cross)_(pt|scratch)(?:_(norm\w*))?_s(\d+)_training$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="output")
    args = ap.parse_args()
    root = args.output_root

    # map training-run basename -> its test report accuracy
    test_acc: dict[str, float] = {}
    for td in glob.glob(os.path.join(root, "*_test")):
        src = os.path.join(td, "checkpoint_source.txt")
        rep = os.path.join(td, "test_report.txt")
        if not (os.path.exists(src) and os.path.exists(rep)):
            continue
        ck = os.path.basename(open(src, encoding="utf-8").read().strip().replace("\\", "/"))
        m = re.search(r"overall accuracy: ([0-9.]+)", open(rep, encoding="utf-8").read())
        if m:
            test_acc[ck] = float(m.group(1))  # later test dirs overwrite earlier ones

    cells: dict[tuple[str, str, str, str], list[tuple[int, float, float]]] = defaultdict(list)
    for run in sorted(glob.glob(os.path.join(root, "*_training"))):
        base = os.path.basename(run)
        m = TAG_RE.match(base)
        if not m:
            continue
        dataset, _, interaction, init, variant, seed = m.groups()
        variant = variant or "raw"
        log = os.path.join(run, "epoch_log.csv")
        if not os.path.exists(log):
            continue
        va = [float(r["val_acc"]) for r in csv.DictReader(open(log, encoding="utf-8"))]
        if not va or base not in test_acc:
            print(f"  (skipping {base}: missing epochs or test result)")
            continue
        cells[(dataset, interaction, init, variant)].append((int(seed), max(va), test_acc[base]))

    if not cells:
        raise SystemExit("No final-comparison runs found (tag pattern <ds>_<mode>_<init>_s<seed>).")

    rows = []
    for (dataset, interaction, init, variant), vals in sorted(cells.items()):
        seeds = sorted(s for s, _, _ in vals)
        v = np.array([x for _, x, _ in vals])
        t = np.array([x for _, _, x in vals])
        rows.append({
            "dataset": dataset, "interaction": interaction, "init": init, "variant": variant,
            "n_seeds": len(vals), "seeds": " ".join(map(str, seeds)),
            "val_mean": v.mean(), "val_std": v.std(ddof=0),
            "test_mean": t.mean(), "test_std": t.std(ddof=0),
            "test_each": " ".join(f"{x:.3f}" for _, _, x in sorted(vals)),
        })

    out_csv = os.path.join(root, "final_comparison_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    print("\n| dataset | interaction | init    | variant | val acc (mean+/-std) | TEST acc (mean+/-std) | per-seed test |")
    print("|---------|-------------|---------|---------|----------------------|-----------------------|---------------|")
    for r in rows:
        print(f"| {r['dataset']:7s} | {r['interaction']:11s} | {r['init']:7s} | {r['variant']:7s} "
              f"| {r['val_mean']:.3f} +/- {r['val_std']:.3f}      "
              f"| {r['test_mean']:.3f} +/- {r['test_std']:.3f}       "
              f"| {r['test_each']} |")
    print(f"\nwritten: {out_csv}")


if __name__ == "__main__":
    main()
