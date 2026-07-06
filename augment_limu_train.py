"""
Augment LIMU-style *train*.json pose sequences (150 frames × 68 values, JSON null for missing).

Reads only files matching *_train.json under --input-root, writes mirrored layout under
--output-root with expanded ``data``: originals first, then ``aug_per_seq`` augmented copies
per original sequence. Defaults: read and write ``LIMU_json_v2/<class>/<class>_train.json``
in-place (same directory as this script). Use ``--backup`` to preserve originals before overwrite.

Augmentations (per augmented copy; geometry sampled once per sequence, applied to every frame):
  flip + COCO-17 LR swap (optional prob), uniform scale, rotation about (0.5,0.5), translation,
  then joint-wise Gaussian jitter (x,y paired) with optional radial cap + clip, keypoint dropout, optional time reversal.

**Training (GNN_new)**

``training.py`` defaults to ``../datasets/LIMU_dataset``. This script writes augmented
JSONs there by default while reading from repo-root ``LIMU_dataset``. Use ``--backup`` when
overwriting existing ``*_train.json`` under the output tree. Override ``--input-root`` /
``--output-root`` if your paths differ.

**Conda / environments (critical)**

- **Never** run ``pip install`` or ``conda install`` into **(base)** — it can break your shared setup.
- This script needs **numpy** only. Create and **activate** a dedicated env first, e.g.::

    conda create -n limu_aug python=3.11
    conda activate limu_aug
    conda install -n limu_aug -c conda-forge numpy

  Or: ``pip install numpy`` only while **limu_aug** (or another non-base env) is active.

- If ``CONDA_DEFAULT_ENV`` is ``base``, this script **exits** unless you pass ``--allow-base``.

Example::

  conda activate limu_aug
  python utility/augment_limu_train.py --seed 0
  python utility/augment_limu_train.py --backup --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "LIMU_json_v2"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "LIMU_json_v2"

# After horizontal mirror x' = 1-x, output joint i takes mirrored coords from input joint SWAP17[i].
SWAP17 = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)

_JSON_KWARGS = {"indent": 2, "ensure_ascii": False}


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _pair_ok(x: Any, y: Any) -> bool:
    return _is_num(x) and _is_num(y)


def decode_frame(f68: list[Any]) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    p1: list[tuple[Any, Any]] = []
    p2: list[tuple[Any, Any]] = []
    for j in range(17):
        b = 4 * j
        p1.append((f68[b], f68[b + 1]))
        p2.append((f68[b + 2], f68[b + 3]))
    return p1, p2


def encode_frame(p1: list[tuple[Any, Any]], p2: list[tuple[Any, Any]]) -> list[Any]:
    out: list[Any] = []
    for j in range(17):
        out.extend([p1[j][0], p1[j][1], p2[j][0], p2[j][1]])
    return out


def flip_swap_person(person: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    """Horizontal flip x -> 1-x plus COCO-17 left/right index swap."""
    out: list[tuple[Any, Any]] = []
    for i in range(17):
        j = SWAP17[i]
        x, y = person[j]
        if not _pair_ok(x, y):
            out.append((None, None))
        else:
            out.append((1.0 - float(x), float(y)))
    return out


def apply_geom_point(
    x: float,
    y: float,
    scale: float,
    cos_a: float,
    sin_a: float,
    tx: float,
    ty: float,
) -> tuple[float, float]:
    xs = scale * (x - 0.5) + 0.5
    ys = scale * (y - 0.5) + 0.5
    xr = cos_a * (xs - 0.5) - sin_a * (ys - 0.5) + 0.5
    yr = sin_a * (xs - 0.5) + cos_a * (ys - 0.5) + 0.5
    return xr + tx, yr + ty


def transform_person(
    person: list[tuple[Any, Any]],
    *,
    do_flip: bool,
    scale: float,
    cos_a: float,
    sin_a: float,
    tx: float,
    ty: float,
) -> list[tuple[Any, Any]]:
    p = flip_swap_person(person) if do_flip else list(person)
    out: list[tuple[Any, Any]] = []
    for x, y in p:
        if not _pair_ok(x, y):
            out.append((None, None))
        else:
            nx, ny = apply_geom_point(float(x), float(y), scale, cos_a, sin_a, tx, ty)
            out.append((nx, ny))
    return out


def noise_clip_frame(
    f68: list[Any],
    rng: np.random.Generator,
    noise_std: float,
    clip: bool,
    cap_mult: float,
) -> list[Any]:
    """Jitter observed keypoints. When cap_mult > 0, limit displacement per joint so rare 2-axis tail noise cannot fling one point far away."""
    out = list(f68)
    if noise_std <= 0.0:
        return out
    # Legacy: independent noise on every scalar (x and y uncorrelated; heavy tails on joint displacement).
    if cap_mult <= 0.0:
        for i in range(len(out)):
            v = out[i]
            if not _is_num(v):
                continue
            v = float(v) + float(rng.normal(0.0, noise_std))
            if clip:
                v = max(0.0, min(1.0, v))
            out[i] = v
        return out

    cap = float(cap_mult) * float(noise_std)
    for j in range(17):
        for person_off in (0, 2):
            b = 4 * j + person_off
            x, y = out[b], out[b + 1]
            if not _pair_ok(x, y):
                continue
            dx, dy = rng.normal(0.0, noise_std, size=2)
            r = math.hypot(float(dx), float(dy))
            if r > cap and r > 0.0:
                s = cap / r
                dx = float(dx) * s
                dy = float(dy) * s
            nx = float(x) + dx
            ny = float(y) + dy
            if clip:
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
            out[b] = nx
            out[b + 1] = ny
    return out


def keypoint_dropout_frame(f68: list[Any], rng: np.random.Generator, p: float) -> list[Any]:
    if p <= 0.0:
        return f68
    out = list(f68)
    for j in range(17):
        b = 4 * j
        for person_off in (0, 2):
            x, y = out[b + person_off], out[b + person_off + 1]
            if _pair_ok(x, y) and rng.random() < p:
                out[b + person_off] = None
                out[b + person_off + 1] = None
    return out


def augment_sequence(
    sequence: list[list[Any]],
    rng: np.random.Generator,
    *,
    flip_prob: float,
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
    s_lo, s_hi = 1.0 - scale_delta, 1.0 + scale_delta
    scale = float(rng.uniform(s_lo, s_hi))
    angle_rad = math.radians(float(rng.uniform(-max_deg, max_deg)))
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    tx = float(rng.uniform(-translate_delta, translate_delta))
    ty = float(rng.uniform(-translate_delta, translate_delta))

    aug_frames: list[list[Any]] = []
    for frame in sequence:
        if len(frame) != 68:
            raise ValueError(f"expected frame length 68, got {len(frame)}")
        p1, p2 = decode_frame(frame)
        p1t = transform_person(p1, do_flip=do_flip, scale=scale, cos_a=cos_a, sin_a=sin_a, tx=tx, ty=ty)
        p2t = transform_person(p2, do_flip=do_flip, scale=scale, cos_a=cos_a, sin_a=sin_a, tx=tx, ty=ty)
        enc = encode_frame(p1t, p2t)
        enc = noise_clip_frame(enc, rng, noise_std, clip, noise_cap_mult)
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
    n_frames = len(seqs[0])
    for si, seq in enumerate(seqs):
        if not isinstance(seq, list):
            raise ValueError(f"{path}: sequence {si} not a list")
        if len(seq) != n_frames:
            raise ValueError(f"{path}: sequence {si} length {len(seq)}, expected {n_frames}")
        for fi, frame in enumerate(seq):
            if not isinstance(frame, list) or len(frame) != 68:
                raise ValueError(f"{path}: seq {si} frame {fi} bad length (want 68)")
    return seqs  # type: ignore[return-value]


def process_file(
    src: Path,
    dst: Path,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> None:
    with open(src, encoding="utf-8") as f:
        payload = json.load(f)

    seqs = validate_payload(payload, src)
    out_data: list[list[list[Any]]] = [list(map(list, s)) for s in seqs]

    for _ in range(args.aug_per_seq):
        for seq in seqs:
            aug = augment_sequence(
                seq,
                rng,
                flip_prob=0.0 if args.no_flip else args.flip_prob,
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
            out_data.append(aug)

    out_payload = {"index": payload.get("index", ""), "data": out_data}
    dst.parent.mkdir(parents=True, exist_ok=True)
    if args.backup and dst.is_file():
        backup = dst.parent / f"{dst.name}.pre_aug_backup"
        shutil.copy2(dst, backup)

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, **_JSON_KWARGS)
        f.write("\n")

    n_orig = len(seqs)
    print(
        f"{src.relative_to(args.input_root)} -> {dst.relative_to(args.output_root)}  "
        f"({n_orig} seqs + {args.aug_per_seq * n_orig} aug)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Augment LIMU *_train.json pose sequences (numpy required). "
            "Do not pip/conda install into conda (base); use a dedicated env (e.g. limu_aug). "
            "Refuses to run in (base) unless --allow-base."
        ),
    )
    ap.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Tree to scan for *_train.json (default: {DEFAULT_INPUT_ROOT})",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Mirror input layout here (default: {DEFAULT_OUTPUT_ROOT})",
    )
    ap.add_argument(
        "--aug-per-seq",
        type=int,
        default=4,
        help="Augmented copies per original sequence (default 4 => 5x total sequences)",
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--backup",
        action="store_true",
        help="If destination exists, copy it to <filename>.pre_aug_backup before overwrite",
    )
    ap.add_argument(
        "--allow-base",
        action="store_true",
        help="Allow running when CONDA_DEFAULT_ENV is base (not recommended)",
    )

    ap.add_argument("--max-deg", type=float, default=4.0, help="Max rotation magnitude (degrees)")
    ap.add_argument(
        "--noise-std",
        type=float,
        default=0.010,
        help="Gaussian noise std (per axis) on normalized coords; paired per keypoint (default 0.010 ~= 67%% of mean inter-frame joint movement)",
    )
    ap.add_argument(
        "--noise-cap-mult",
        type=float,
        default=2.5,
        help="Max joint displacement radius = this * noise_std (0 = legacy independent noise per scalar); default cap = 0.025 at noise_std=0.010",
    )
    ap.add_argument("--no-clip", action="store_true", help="Do not clip coords to [0,1] after noise")

    ap.add_argument("--flip-prob", type=float, default=0.5, help="Probability of hflip+LR swap per aug copy")
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--scale-delta", type=float, default=0.05, help="Scale ~ U(1-d, 1+d) per aug copy")
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--translate-delta", type=float, default=0.03, help="Uniform shift per axis")
    ap.add_argument("--no-translate", action="store_true")

    ap.add_argument("--kp-dropout-prob", type=float, default=0.02, help="Per observed joint-person pair")
    ap.add_argument("--allow-time-reverse", action="store_true")
    ap.add_argument("--time-reverse-prob", type=float, default=0.5, help="If allow-time-reverse, prob to reverse")

    args = ap.parse_args()
    if args.aug_per_seq < 0:
        raise SystemExit("--aug-per-seq must be >= 0")

    if os.environ.get("CONDA_DEFAULT_ENV") == "base" and not args.allow_base:
        print(
            "Refusing to run in conda (base): create/activate a dedicated env and install numpy there, "
            "or pass --allow-base if you accept the risk.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    args.input_root = input_root
    args.output_root = output_root
    if not input_root.is_dir():
        raise SystemExit(f"Not a directory: {input_root}")

    rng = np.random.default_rng(args.seed)

    train_files = sorted(input_root.rglob("*_train.json"))
    if not train_files:
        raise SystemExit(f"No *_train.json under {input_root}")

    for src in train_files:
        try:
            rel = src.relative_to(input_root)
        except ValueError:
            rel = Path(src.name)
        dst = output_root / rel
        try:
            process_file(src, dst, rng, args)
        except (ValueError, OSError, json.JSONDecodeError) as e:
            raise SystemExit(f"{src}: {e}") from e

    print(f"Done. Wrote {len(train_files)} file(s) under {output_root}")


if __name__ == "__main__":
    main()
