"""
analyze_camera_angle.py -- accuracy vs camera viewing angle.

Joins per-clip test predictions (test.py's per_clip_predictions.csv, poolable over
several runs/seeds) with the per-clip camera angle (camera_angle.csv from the
extraction workspace) and reports test accuracy per angle bin -- overall and per true
class -- to show how much recognition degrades toward occlusion-heavy viewpoints
(0 deg = camera in line with the two athletes, 90 deg = perpendicular).

  python analyze_camera_angle.py output/20260729_*_test/per_clip_predictions.csv
  python analyze_camera_angle.py --predictions "output/*_test/per_clip_predictions.csv"

Writes output/camera_angle_analysis.csv and output/camera_angle_analysis.png.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_BINS = [0, 15, 30, 45, 60, 75, 90]


def load_angles(path: str) -> dict[str, float]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["clip_id"].strip(): float(r["angle_median"]) for r in csv.DictReader(f)}


def load_predictions(patterns: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pat in patterns:
        files = sorted(glob.glob(pat)) or [pat]
        for p in files:
            with open(p, newline="", encoding="utf-8") as f:
                got = list(csv.DictReader(f))
            print(f"  {p}: {len(got)} predictions")
            rows += got
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Test accuracy vs camera viewing angle.")
    ap.add_argument("predictions", nargs="*",
                    default=["output/*_test/per_clip_predictions.csv"],
                    help="per_clip_predictions.csv path(s) or glob(s); multiple files are "
                         "pooled (e.g. 3 seeds of the same cell)")
    ap.add_argument("--camera-angle-csv", default="camera_angle.csv")
    ap.add_argument("--bins", type=float, nargs="+", default=DEFAULT_BINS,
                    help=f"angle bin edges in degrees (default {DEFAULT_BINS})")
    ap.add_argument("--output-root", default="output")
    args = ap.parse_args()

    angles = load_angles(args.camera_angle_csv)
    rows = load_predictions(args.predictions)
    if not rows:
        raise SystemExit("No predictions found.")

    joined = [(angles[r["clip_id"]], r) for r in rows if r.get("clip_id") in angles]
    n_missing = len(rows) - len(joined)
    if n_missing:
        print(f"WARNING: {n_missing}/{len(rows)} predictions have no camera angle "
              f"(missing clip_id or not in {args.camera_angle_csv}); excluded")
    if not joined:
        raise SystemExit("No predictions could be joined with camera angles.")

    edges = np.asarray(args.bins, float)
    classes = sorted({r["true_class"] for _, r in joined})

    def bin_stats(subset):
        """per-bin (n, accuracy) for a list of (angle, row)."""
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [r for a, r in subset if lo <= a < hi or (hi == edges[-1] and a == hi)]
            n = len(sel)
            acc = np.mean([int(r["correct"]) for r in sel]) if n else np.nan
            out.append((n, acc))
        return out

    overall = bin_stats(joined)
    per_class = {c: bin_stats([(a, r) for a, r in joined if r["true_class"] == c])
                 for c in classes}

    os.makedirs(args.output_root, exist_ok=True)
    out_csv = os.path.join(args.output_root, "camera_angle_analysis.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "bin_lo", "bin_hi", "n", "accuracy"])
        for (lo, hi), (n, acc) in zip(zip(edges[:-1], edges[1:]), overall):
            w.writerow(["overall", lo, hi, n, f"{acc:.4f}" if n else ""])
        for c in classes:
            for (lo, hi), (n, acc) in zip(zip(edges[:-1], edges[1:]), per_class[c]):
                w.writerow([c, lo, hi, n, f"{acc:.4f}" if n else ""])

    # correlation-style summary
    a_all = np.array([a for a, _ in joined])
    c_all = np.array([int(r["correct"]) for _, r in joined])
    print(f"\n{len(joined)} joined predictions | angle range "
          f"{a_all.min():.1f}-{a_all.max():.1f} deg | overall acc {c_all.mean():.3f}")
    print(f"{'bin':>12s} {'n':>5s} {'acc':>6s}")
    for (lo, hi), (n, acc) in zip(zip(edges[:-1], edges[1:]), overall):
        print(f"{lo:5.0f}-{hi:3.0f} deg {n:5d} {acc if n else float('nan'):6.3f}")

    # ---- figure: overall + per-class accuracy vs angle, and the angle distribution ----
    centers = (edges[:-1] + edges[1:]) / 2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), height_ratios=[3, 1], sharex=True)
    accs = [acc for _, acc in overall]
    ax1.plot(centers, accs, "o-", color="black", lw=2, label="overall", zorder=3)
    for c in classes:
        ax1.plot(centers, [acc for _, acc in per_class[c]], "o--", ms=4, lw=1, alpha=0.7, label=c)
    ax1.set_ylabel("test accuracy")
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7, ncols=2)
    ax1.set_title("Accuracy vs camera angle (0 = in-line/occluded, 90 = perpendicular)")
    ax2.hist(a_all, bins=edges, color="tab:blue", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("camera angle [deg]")
    ax2.set_ylabel("n preds")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(args.output_root, "camera_angle_analysis.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\nwritten: {out_csv}\n         {out_png}")


if __name__ == "__main__":
    main()
