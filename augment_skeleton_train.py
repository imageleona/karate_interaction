"""
Augment two-person karate skeleton *_train.json sequences (120 frames x 68 values).

This is the karate-dataset counterpart of ``augment_limu_train.py``. It keeps that script's
structure (scan ``*_train.json``, mirror the directory tree, write originals first then
``aug_per_seq`` augmented copies per original sequence, per-sequence sampled geometry, numpy
only, conda-``base`` guard) but is adapted to THIS dataset's contract:

  * Frame = 68 floats in **two contiguous 34-blocks**: person1 = indices 0..33,
    person2 = 34..67; within a person joint ``k`` (COCO-17) is ``row[p*34 + k*2]`` (x),
    ``+1`` (y). (The LIMU script uses a different per-joint interleaved layout.)
  * Coordinates are **raw pixels** (~1920x1080), NOT normalized [0,1]. Flip mirrors about the
    frame width, scale/rotation are about the image centre, noise/clip are in pixel units.
  * Missing keypoints are marked as BOTH ``null`` and ``0.0``; both are treated as missing and
    written back as ``0.0``.

By default reads ``skeleton_dataset/`` and writes a NEW parallel tree
``skeleton_dataset_augmented/`` (originals untouched). Non-train files (``*_test.json`` and
``class_names.json``) are copied through verbatim so the output tree is self-contained and can
be used directly as a ``--data-dir`` for training AND testing.

Augmentations (per augmented copy; geometry sampled once per sequence, applied to every frame):
  flip about frame width + COCO-17 LR joint swap (+ optional person1<->person2 swap),
  then per-keypoint Gaussian jitter (x,y paired) with radial cap + clip to frame, and keypoint
  dropout. Scale / rotation / translation are OFF by default (they blur the vertical-height cue
  separating middle vs high kicks); re-enable via --scale-delta/--max-deg/--translate-delta.
  Time reversal off by default (a reversed strike is not a valid sample).

Train/val split: if <input-root>/split_report.csv exists (or --split-report is given), the val
split is SOURCE-GROUPED -- whole source videos are held out (consistently across the 4class and
8class labelings), so val measures generalization to unseen footage. The chosen val sources are
recorded in <output-root>/val_split.json. Without a report it falls back to a random clip split.

**Conda / environments (critical)**

- **Never** ``pip``/``conda install`` into **(base)** -- it can break your shared setup.
- This script needs **numpy** only. Activate a dedicated env first (e.g. ``limu_aug``).
- If ``CONDA_DEFAULT_ENV`` is ``base``, this script **exits** unless you pass ``--allow-base``.

Example::

  conda activate limu_aug
  python augment_skeleton_train.py --seed 0
  python augment_skeleton_train.py --seed 0 --aug-per-seq 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "skeleton_dataset"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "skeleton_dataset_augmented"

NUM_JOINTS = 17          # COCO-17
PERSON_STRIDE = 34       # 17 joints * 2 coords per person
FRAME_LEN = 68           # 2 persons * 34

# After a horizontal mirror, output joint i takes the mirrored coords of input joint SWAP17[i]
# (COCO-17 left/right swap: eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).
SWAP17 = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)

# Internal missing sentinel for a keypoint (x, y) pair.
MISSING: tuple[Any, Any] = (None, None)

_JSON_KWARGS = {"indent": 2, "ensure_ascii": False}


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _pair_missing(x: Any, y: Any) -> bool:
    """A keypoint is missing if either coord is non-numeric/non-finite, or it is exactly (0, 0)."""
    if not (_is_num(x) and _is_num(y)):
        return True
    return float(x) == 0.0 and float(y) == 0.0


def decode_frame(f68: list[Any]) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    """Flat 68-list -> (person1, person2), each a list of 17 (x, y) tuples (MISSING for absent)."""
    persons: list[list[tuple[Any, Any]]] = []
    for p in range(2):
        base = p * PERSON_STRIDE
        person: list[tuple[Any, Any]] = []
        for k in range(NUM_JOINTS):
            x, y = f68[base + 2 * k], f68[base + 2 * k + 1]
            person.append(MISSING if _pair_missing(x, y) else (float(x), float(y)))
        persons.append(person)
    return persons[0], persons[1]


def encode_frame(p1: list[tuple[Any, Any]], p2: list[tuple[Any, Any]]) -> list[float]:
    """(person1, person2) -> flat 68-list; missing keypoints written as 0.0."""
    out: list[float] = []
    for person in (p1, p2):
        for x, y in person:
            if x is None or y is None:
                out.extend([0.0, 0.0])
            else:
                out.extend([float(x), float(y)])
    return out


def flip_swap_person(person: list[tuple[Any, Any]], width: float) -> list[tuple[Any, Any]]:
    """Horizontal flip x -> width - x plus COCO-17 left/right index swap."""
    out: list[tuple[Any, Any]] = []
    for i in range(NUM_JOINTS):
        x, y = person[SWAP17[i]]
        out.append(MISSING if x is None else (width - float(x), float(y)))
    return out


def apply_geom_point(
    x: float,
    y: float,
    *,
    cx: float,
    cy: float,
    scale: float,
    cos_a: float,
    sin_a: float,
    tx: float,
    ty: float,
) -> tuple[float, float]:
    """Scale then rotate about (cx, cy), then translate by (tx, ty). All in pixel units."""
    xs = scale * (x - cx)
    ys = scale * (y - cy)
    xr = cos_a * xs - sin_a * ys + cx
    yr = sin_a * xs + cos_a * ys + cy
    return xr + tx, yr + ty


def transform_person(
    person: list[tuple[Any, Any]],
    **geom: float,
) -> list[tuple[Any, Any]]:
    out: list[tuple[Any, Any]] = []
    for x, y in person:
        if x is None:
            out.append(MISSING)
        else:
            out.append(apply_geom_point(float(x), float(y), **geom))
    return out


def noise_clip_frame(
    f68: list[Any],
    rng: np.random.Generator,
    *,
    noise_std: float,
    clip: bool,
    cap_mult: float,
    width: float,
    height: float,
) -> list[Any]:
    """Jitter observed keypoints in pixel space; cap displacement radius; optionally clip to frame."""
    if noise_std <= 0.0:
        return list(f68)
    out = list(f68)
    cap = float(cap_mult) * float(noise_std) if cap_mult > 0.0 else None
    for p in range(2):
        base = p * PERSON_STRIDE
        for k in range(NUM_JOINTS):
            xi, yi = base + 2 * k, base + 2 * k + 1
            x, y = out[xi], out[yi]
            if _pair_missing(x, y):
                continue
            dx, dy = rng.normal(0.0, noise_std, size=2)
            if cap is not None:
                r = math.hypot(float(dx), float(dy))
                if r > cap and r > 0.0:
                    s = cap / r
                    dx, dy = float(dx) * s, float(dy) * s
            nx, ny = float(x) + float(dx), float(y) + float(dy)
            if clip:
                nx = min(max(nx, 0.0), width)
                ny = min(max(ny, 0.0), height)
            out[xi], out[yi] = nx, ny
    return out


def keypoint_dropout_frame(f68: list[Any], rng: np.random.Generator, p: float) -> list[Any]:
    if p <= 0.0:
        return f68
    out = list(f68)
    for person in range(2):
        base = person * PERSON_STRIDE
        for k in range(NUM_JOINTS):
            xi, yi = base + 2 * k, base + 2 * k + 1
            if not _pair_missing(out[xi], out[yi]) and rng.random() < p:
                out[xi], out[yi] = 0.0, 0.0
    return out


def augment_sequence(
    sequence: list[list[Any]],
    rng: np.random.Generator,
    *,
    width: float,
    height: float,
    flip_prob: float,
    flip_person_swap: bool,
    scale_delta: float,
    max_deg: float,
    translate_delta: float,
    noise_std: float,
    clip: bool,
    noise_cap_mult: float,
    kp_dropout_prob: float,
    allow_time_reverse: bool,
    time_reverse_prob: float,
) -> list[list[Any]]:
    do_flip = flip_prob > 0.0 and rng.random() < flip_prob
    scale = float(rng.uniform(1.0 - scale_delta, 1.0 + scale_delta))
    angle_rad = math.radians(float(rng.uniform(-max_deg, max_deg)))
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    tx = float(rng.uniform(-translate_delta, translate_delta)) * width
    ty = float(rng.uniform(-translate_delta, translate_delta)) * height
    geom = dict(cx=width / 2.0, cy=height / 2.0, scale=scale, cos_a=cos_a, sin_a=sin_a, tx=tx, ty=ty)

    aug_frames: list[list[Any]] = []
    for frame in sequence:
        if len(frame) != FRAME_LEN:
            raise ValueError(f"expected frame length {FRAME_LEN}, got {len(frame)}")
        p1, p2 = decode_frame(frame)
        if do_flip:
            p1 = flip_swap_person(p1, width)
            p2 = flip_swap_person(p2, width)
            if flip_person_swap:
                p1, p2 = p2, p1  # keep "left person = P1" after the mirror
        p1 = transform_person(p1, **geom)
        p2 = transform_person(p2, **geom)
        enc = encode_frame(p1, p2)
        enc = noise_clip_frame(
            enc, rng, noise_std=noise_std, clip=clip, cap_mult=noise_cap_mult,
            width=width, height=height,
        )
        enc = keypoint_dropout_frame(enc, rng, kp_dropout_prob)
        aug_frames.append(enc)

    if allow_time_reverse and time_reverse_prob > 0.0 and rng.random() < time_reverse_prob:
        aug_frames.reverse()

    return aug_frames


def validate_payload(data: Any, path: Path) -> list[list[list[Any]]]:
    if not isinstance(data, dict) or "data" not in data:
        raise ValueError(f"{path}: missing 'data'")
    seqs = data["data"]
    if not isinstance(seqs, list) or not seqs:
        raise ValueError(f"{path}: empty or invalid 'data'")
    for si, seq in enumerate(seqs):
        if not isinstance(seq, list) or not seq:
            raise ValueError(f"{path}: sequence {si} not a non-empty list")
        for fi, frame in enumerate(seq):
            if not isinstance(frame, list) or len(frame) != FRAME_LEN:
                raise ValueError(f"{path}: seq {si} frame {fi} bad length (want {FRAME_LEN})")
    return seqs  # type: ignore[return-value]


def load_split_report(path: Path) -> list[dict[str, str]] | None:
    """Load split_report.csv rows (clip_id, technique, point, source, split, ...)."""
    if not path.is_file():
        return None
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    needed = {"technique", "point", "source", "split"}
    if not rows or not needed.issubset(rows[0].keys()):
        return None
    return rows


def source_by_clip_id(rows: list[dict[str, str]]) -> dict[str, str]:
    """clip_id -> source video map from split_report.csv (for JSONs that carry clip_ids)."""
    return {r["clip_id"]: r["source"] for r in rows if r.get("clip_id")}


def class_train_sources(rows: list[dict[str, str]], cls: str) -> list[str] | None:
    """Ordered source list for class ``cls``'s train rows (CSV order == JSON sequence order).

    Positional FALLBACK for JSONs without ``clip_ids``. ``cls`` is the filename stem before
    ``_train.json``: either a bare technique (4class) or ``<technique>_point`` /
    ``<technique>_no_point`` (8class). Finer class names (15class) are not parsed here --
    those trees carry ``clip_ids`` and use :func:`source_by_clip_id` instead.
    """
    if cls.endswith("_no_point"):
        tech, point = cls[: -len("_no_point")], "0"
    elif cls.endswith("_point"):
        tech, point = cls[: -len("_point")], "1"
    else:
        tech, point = cls, None
    out = [r["source"] for r in rows
           if r["technique"] == tech and r["split"] == "train"
           and (point is None or r["point"] == point)]
    return out or None


def build_source_val_split(
    rows: list[dict[str, str]],
    val_ratio: float,
    split_rng: np.random.Generator,
) -> set[str]:
    """Pick val SOURCE VIDEOS (whole sources, per technique) totalling ~val_ratio of train clips.

    The assignment is per source, so it is automatically consistent across the 4class and
    8class labelings (every source contains exactly one technique).
    """
    val_sources: set[str] = set()
    if val_ratio <= 0.0:
        return val_sources
    by_tech: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r["split"] == "train":
            by_tech[r["technique"]].append(r)
    for tech in sorted(by_tech):
        trows = by_tech[tech]
        counts = Counter(r["source"] for r in trows)
        sources = sorted(counts)
        if len(sources) < 2:
            print(f"WARNING: technique {tech} has {len(sources)} train source(s); "
                  f"cannot hold one out for val")
            continue
        target = val_ratio * len(trows)
        order = split_rng.permutation(len(sources))
        taken = 0
        for k in order:
            s = sources[int(k)]
            if taken >= target:
                break
            # never move the last remaining train source to val
            if sum(1 for x in sources if x not in val_sources and x != s) == 0:
                continue
            val_sources.add(s)
            taken += counts[s]
        clips_val = sum(counts[s] for s in sources if s in val_sources)
        held = sorted(s for s in sources if s in val_sources)
        print(f"  val sources [{tech}]: {held}  "
              f"({clips_val}/{len(trows)} clips = {clips_val / len(trows):.0%})")
    return val_sources


def _write_payload(index: Any, data: list, dst: Path, backup: bool,
                   clip_ids: list[str] | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if backup and dst.is_file():
        shutil.copy2(dst, dst.parent / f"{dst.name}.pre_aug_backup")
    payload: dict[str, Any] = {"index": index, "data": data}
    if clip_ids is not None:
        payload["clip_ids"] = clip_ids
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(payload, f, **_JSON_KWARGS)
        f.write("\n")


def process_train_file(
    src: Path,
    dst: Path,
    rng: np.random.Generator,
    split_rng: np.random.Generator,
    args: argparse.Namespace,
    seq_sources: list[str] | None = None,
    val_sources: set[str] | None = None,
    source_by_clip: dict[str, str] | None = None,
) -> None:
    with open(src, encoding="utf-8") as f:
        payload = json.load(f)

    seqs = validate_payload(payload, src)
    index = payload.get("index", "")
    n = len(seqs)

    clip_ids: list[str] | None = payload.get("clip_ids")
    if clip_ids is not None and len(clip_ids) != n:
        print(f"WARNING: {src.name}: clip_ids ({len(clip_ids)}) != sequences ({n}); ignoring them")
        clip_ids = None

    # Per-sequence source videos: prefer the explicit clip_id join, fall back to the
    # positional CSV-order convention for older files without clip_ids.
    if clip_ids is not None and source_by_clip is not None:
        by_id = [source_by_clip.get(cid) for cid in clip_ids]
        if all(s is not None for s in by_id):
            seq_sources = by_id  # type: ignore[assignment]
        else:
            missing = [cid for cid, s in zip(clip_ids, by_id) if s is None]
            print(f"WARNING: {src.name}: {len(missing)} clip_ids not in split_report "
                  f"({missing[:3]}...); using positional source mapping")

    # ----- split ORIGINAL clips into train/val BEFORE augmenting (prevents leakage) -----
    # Preferred: SOURCE-GROUPED split -- all clips from one source video stay on one side,
    # so val measures generalization to unseen videos, not recall of seen ones.
    split_kind = "random-clip"
    if seq_sources is not None and val_sources is not None and len(seq_sources) == n:
        val_ids = {i for i, s in enumerate(seq_sources) if s in val_sources}
        split_kind = "source-grouped"
    else:
        if seq_sources is not None and len(seq_sources) != n:
            print(f"WARNING: {src.name}: split_report rows ({len(seq_sources)}) != sequences ({n}); "
                  f"falling back to random clip split")
        if args.val_ratio > 0.0 and n >= 2:
            n_val = min(n - 1, max(1, int(round(n * args.val_ratio))))
        else:
            n_val = 0
        perm = split_rng.permutation(n)
        val_ids = set(int(i) for i in perm[:n_val])
    train_seqs = [seqs[i] for i in range(n) if i not in val_ids]
    val_seqs = [seqs[i] for i in range(n) if i in val_ids]
    train_ids = [clip_ids[i] for i in range(n) if i not in val_ids] if clip_ids else None
    val_ids_list = [clip_ids[i] for i in range(n) if i in val_ids] if clip_ids else None

    # ----- augment ONLY the train split (originals first, then aug_per_seq copies) -----
    out_train: list[list[list[Any]]] = [list(map(list, s)) for s in train_seqs]
    out_train_ids = list(train_ids) if train_ids is not None else None
    for round_i in range(args.aug_per_seq):
        if out_train_ids is not None:
            out_train_ids += [f"{cid}#aug{round_i}" for cid in train_ids]
        for seq in train_seqs:
            out_train.append(
                augment_sequence(
                    seq,
                    rng,
                    width=args.frame_width,
                    height=args.frame_height,
                    flip_prob=0.0 if args.no_flip else args.flip_prob,
                    flip_person_swap=not args.no_flip_person_swap,
                    scale_delta=0.0 if args.no_scale else args.scale_delta,
                    max_deg=args.max_deg,
                    translate_delta=0.0 if args.no_translate else args.translate_delta,
                    noise_std=args.noise_std,
                    clip=not args.no_clip,
                    noise_cap_mult=args.noise_cap_mult,
                    kp_dropout_prob=args.kp_dropout_prob,
                    allow_time_reverse=args.allow_time_reverse,
                    time_reverse_prob=args.time_reverse_prob,
                )
            )

    _write_payload(index, out_train, dst, args.backup, clip_ids=out_train_ids)

    # ----- val = CLEAN, un-augmented originals, disjoint from everything in train -----
    if val_seqs:
        val_dst = dst.parent / src.name.replace("_train.json", "_val.json")
        out_val = [list(map(list, s)) for s in val_seqs]
        _write_payload(index, out_val, val_dst, args.backup, clip_ids=val_ids_list)

    print(
        f"{src.relative_to(args.input_root)}  ->  "
        f"train {len(train_seqs)}+{args.aug_per_seq * len(train_seqs)}aug={len(out_train)}  |  "
        f"val {len(val_seqs)} (clean, {split_kind})"
    )


def copy_through(src: Path, dst: Path, args: argparse.Namespace) -> None:
    if src.resolve() == dst.resolve():
        return  # in-place: nothing to copy
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Augment two-person karate skeleton *_train.json (numpy required). "
            "Refuses to run in conda (base) unless --allow-base."
        ),
    )
    ap.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT,
                    help=f"Tree to scan for *_train.json (default: {DEFAULT_INPUT_ROOT})")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                    help=f"Mirror input layout here (default: {DEFAULT_OUTPUT_ROOT})")
    ap.add_argument("--aug-per-seq", type=int, default=4,
                    help="Augmented copies per original sequence (default 4 => 5x total)")
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="Fraction of ORIGINAL clips per class held out as a clean validation set "
                         "(written to <class>_val.json; NOT augmented). 0 disables. Default 0.2")
    ap.add_argument("--split-report", type=Path, default=None,
                    help="split_report.csv with clip->source mapping; enables SOURCE-GROUPED "
                         "train/val split (default: <input-root>/split_report.csv). Whole source "
                         "videos go to val so train/val share no footage.")
    ap.add_argument("--no-source-split", action="store_true",
                    help="Force the old random per-clip val split even if split_report.csv exists")
    ap.add_argument("--split-seed", type=int, default=42,
                    help="Seed for the train/val split of original clips (reproducible). Default 42")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--backup", action="store_true",
                    help="If a destination train file exists, copy it to <name>.pre_aug_backup first")
    ap.add_argument("--no-copy-aux", action="store_true",
                    help="Do not copy *_test.json / class_names.json into the output tree")
    ap.add_argument("--allow-base", action="store_true",
                    help="Allow running when CONDA_DEFAULT_ENV is base (not recommended)")

    ap.add_argument("--frame-width", type=float, default=1920.0, help="Pixel frame width (flip axis / clip)")
    ap.add_argument("--frame-height", type=float, default=1080.0, help="Pixel frame height (clip)")

    # Geometry defaults are 0 (flip + noise + dropout only): scale/translate/rotate perturb the
    # vertical-height cue that separates chudan-geri from jodan-geri (middle vs high kick) and
    # hurt test accuracy. Pass e.g. --scale-delta 0.05 --translate-delta 0.03 --max-deg 4 to re-enable.
    ap.add_argument("--max-deg", type=float, default=0.0, help="Max rotation magnitude in degrees (default 0 = off)")
    ap.add_argument("--noise-std", type=float, default=6.0,
                    help="Gaussian noise std in PIXELS (per axis), paired per keypoint (default 6.0)")
    ap.add_argument("--noise-cap-mult", type=float, default=2.5,
                    help="Max joint displacement radius = this * noise_std (0 = uncapped)")
    ap.add_argument("--no-clip", action="store_true", help="Do not clip coords to the frame after noise")

    ap.add_argument("--flip-prob", type=float, default=0.5, help="Probability of hflip+LR swap per aug copy")
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--no-flip-person-swap", action="store_true",
                    help="Do NOT swap P1<->P2 after a horizontal flip (default: swap to keep left=P1)")
    ap.add_argument("--scale-delta", type=float, default=0.0,
                    help="Scale ~ U(1-d, 1+d) per aug copy (default 0 = off; see geometry note above)")
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--translate-delta", type=float, default=0.0,
                    help="Shift per axis as a fraction of frame size (default 0 = off; see geometry note)")
    ap.add_argument("--no-translate", action="store_true")

    ap.add_argument("--kp-dropout-prob", type=float, default=0.02, help="Per observed keypoint per frame")
    ap.add_argument("--allow-time-reverse", action="store_true")
    ap.add_argument("--time-reverse-prob", type=float, default=0.5)

    args = ap.parse_args()
    if args.aug_per_seq < 0:
        raise SystemExit("--aug-per-seq must be >= 0")

    if os.environ.get("CONDA_DEFAULT_ENV") == "base" and not args.allow_base:
        print(
            "Refusing to run in conda (base): activate a dedicated env (e.g. limu_aug) with numpy, "
            "or pass --allow-base if you accept the risk.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    args.input_root, args.output_root = input_root, output_root
    if not input_root.is_dir():
        raise SystemExit(f"Not a directory: {input_root}")

    rng = np.random.default_rng(args.seed)
    split_rng = np.random.default_rng(args.split_seed)  # independent, always-reproducible split

    # ----- source-grouped val split (preferred): whole source videos held out for val -----
    report_rows: list[dict[str, str]] | None = None
    val_sources: set[str] | None = None
    if not args.no_source_split:
        report_path = args.split_report or (input_root / "split_report.csv")
        report_rows = load_split_report(report_path)
        if report_rows is None:
            print(f"NOTE: no usable split report at {report_path}; using random per-clip val split "
                  f"(val may share source videos with train).")
        else:
            print(f"Source-grouped val split from {report_path}:")
            val_sources = build_source_val_split(report_rows, args.val_ratio, split_rng)

    train_files = sorted(input_root.rglob("*_train.json"))
    if not train_files:
        raise SystemExit(f"No *_train.json under {input_root}")

    src_by_id = source_by_clip_id(report_rows) if report_rows is not None else None
    for src in train_files:
        rel = src.relative_to(input_root)
        seq_sources = None
        if report_rows is not None and val_sources is not None:
            seq_sources = class_train_sources(report_rows, src.name[: -len("_train.json")])
        try:
            process_train_file(src, output_root / rel, rng, split_rng, args,
                               seq_sources=seq_sources, val_sources=val_sources,
                               source_by_clip=src_by_id)
        except (ValueError, OSError, json.JSONDecodeError) as e:
            raise SystemExit(f"{src}: {e}") from e

    if val_sources:
        prov = output_root / "val_split.json"
        with open(prov, "w", encoding="utf-8") as f:
            json.dump({"split": "source-grouped", "split_seed": args.split_seed,
                       "val_ratio": args.val_ratio, "val_sources": sorted(val_sources)},
                      f, indent=2)
        print(f"val source list written to {prov}")

    n_aux = 0
    if not args.no_copy_aux:
        aux_files = sorted(input_root.rglob("*_test.json")) + sorted(input_root.rglob("class_names.json"))
        for src in aux_files:
            rel = src.relative_to(input_root)
            copy_through(src, output_root / rel, args)
            n_aux += 1

    print(f"Done. Wrote {len(train_files)} train file(s) + copied {n_aux} aux file(s) under {output_root}")


if __name__ == "__main__":
    main()
