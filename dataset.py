"""
dataset.py -- PyTorch Dataset for two-person COCO-17 skeleton clips.

Implements section 3 of ``YOLO17_WORKSPACE_SPEC.md`` for THIS workspace's data:
  * JSON files ``<class>_{train,test}.json`` = ``{"index": <1-based label>, "data": [seq, ...]}``.
  * frame = 68 floats, two contiguous 34-blocks (person1 = 0..33, person2 = 34..67).
  * missing joints appear as ``null`` and/or ``0.0`` -> zeroed at load.
  * every sample tensor is ``(C, T, V) = (2, num_frames, 34)``.

Unlike the spec's hardcoded ``CLASS_NAMES``, class labels are read from ``class_names.json`` in
the data dir via :func:`load_class_names`, so the same code trains either the 4-class or the
8-class label set just by changing ``--data-dir``.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

INPUT_CHANNELS = 2          # (x, y); set 3 if frame rows ever carry confidence
DEFAULT_NUM_FRAMES = 120    # this dataset's clips are already a fixed 120 frames
JOINTS_PER_PERSON = 17      # COCO-17
NUM_NODES = 34              # two persons

# Fallback if no class_names.json is found (4-class technique set).
DEFAULT_CLASS_NAMES = ["kizami", "chudan", "chudankeri", "jodankeri"]


def load_class_names(data_dir: str) -> list[str]:
    """Read the ordered class list from ``class_names.json`` under ``data_dir`` (recursive).

    The list index + 1 must equal the JSON ``index`` field of the matching class files.
    """
    matches = sorted(glob.glob(os.path.join(data_dir, "**", "class_names.json"), recursive=True))
    if matches:
        with open(matches[0], encoding="utf-8") as f:
            names = json.load(f)
        if isinstance(names, list) and names:
            return [str(n) for n in names]
    return list(DEFAULT_CLASS_NAMES)


def split_sequences_from_raw_data(raw):
    """A raw ``data`` field may be a single sequence or a list of sequences -> normalize to a list."""
    if not isinstance(raw, list) or not raw:
        return []
    if isinstance(raw[0], list) and raw[0] and isinstance(raw[0][0], list):
        return raw          # already list-of-sequences (3 levels deep)
    return [raw]            # single sequence


def extract_two_person_xy_flat(seq: np.ndarray) -> Optional[np.ndarray]:
    """Flat frame array ``(frames, feat_dim)`` -> ``(frames, 34, 2)`` (persons concatenated)."""
    frames, feat_dim = seq.shape
    if feat_dim % INPUT_CHANNELS:
        return None
    detected = feat_dim // INPUT_CHANNELS
    tmp = seq.reshape(frames, detected, INPUT_CHANNELS)
    if detected == NUM_NODES:                       # 34 = 17 joints x 2 persons
        return np.concatenate([tmp[:, 0:JOINTS_PER_PERSON], tmp[:, JOINTS_PER_PERSON:NUM_NODES]], axis=1)
    if detected == JOINTS_PER_PERSON:               # single person
        return tmp
    return None


def pad_resample_time(arr: np.ndarray, num_frames: int) -> np.ndarray:
    """Pad short clips with zero frames; linspace-subsample long clips. arr: (T, ...)."""
    t = arr.shape[0]
    if t == num_frames:
        return arr.astype(np.float32)
    if t < num_frames:
        pad = np.zeros((num_frames - t,) + arr.shape[1:], np.float32)
        return np.vstack((arr.astype(np.float32), pad))
    idx = np.linspace(0, t - 1, num_frames).astype(int)
    return arr[idx].astype(np.float32)


class SkeletonDataset(Dataset):
    """Loads ``*<mode>.json`` under ``data_dir`` into ``(C, T, V)`` tensors + integer labels."""

    def __init__(
        self,
        data_dir: str,
        class_names: list[str],
        mode: str = "train",
        num_frames: int = DEFAULT_NUM_FRAMES,
        do_center=False,
        do_scale: bool = False,
    ):
        assert mode in ("train", "val", "test"), mode
        self.data_dir = data_dir
        self.class_names = list(class_names)
        self.mode = mode
        self.num_frames = num_frames
        # do_center: False | "person" | "scene"  (True kept as alias for "person")
        if do_center is True:
            do_center = "person"
        assert do_center in (False, None, "", "person", "scene"), do_center
        self.do_center = do_center or False
        self.do_scale = do_scale

        self.data_list: list[np.ndarray] = []
        self.labels: list[int] = []
        self._load()

        if not self.data_list:
            raise RuntimeError(f"No {mode} samples found under {data_dir}")
        # self-configure channel count (2 or 3) from the actual data
        self.in_channels = self.data_list[0].shape[0]

    def _resolve_label(self, content: dict, path: str) -> Optional[int]:
        idx = content.get("index", None)
        if idx is not None:
            try:
                label = int(idx) - 1
            except (TypeError, ValueError):
                label = None
            if label is not None and 0 <= label < len(self.class_names):
                return label
        # fallback: first class name found in the filename
        name = os.path.basename(path).lower()
        for i, cn in enumerate(self.class_names):
            if cn.lower() in name:
                return i
        return None

    def _process_sequence(self, seq_raw) -> Optional[np.ndarray]:
        arr = np.array(seq_raw, dtype=np.float32)
        if arr.ndim != 2:
            return None
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)   # null -> nan -> 0.0
        xy = extract_two_person_xy_flat(arr)                        # (T, 34, 2)
        if xy is None:
            return None
        if self.do_center:
            xy = self._center(xy, self.do_center)
        if self.do_scale:
            xy = self._scale(xy)
        xy = pad_resample_time(xy, self.num_frames)                # (T, 34, 2)
        return np.transpose(xy, (2, 0, 1))                         # (C, T, V) = (2, T, 34)

    @staticmethod
    def _center(xy: np.ndarray, mode: str) -> np.ndarray:
        """Subtract a hip-based anchor per frame. mode='person': each person centred on the
        midpoint of their OWN observed hips (encodes own-body-relative pose, e.g. kick height,
        but discards inter-person geometry). mode='scene': both persons centred on the mean of
        the available per-person hip midpoints (removes camera translation, KEEPS the relative
        position between the two fighters). COCO hips = joints 11 & 12; only observed hips are
        averaged. Missing keypoints stay 0."""
        n_persons = xy.shape[1] // JOINTS_PER_PERSON
        out = xy.copy()
        centres, valids = [], []
        for p in range(n_persons):
            base = p * JOINTS_PER_PERSON
            hips = xy[:, [base + 11, base + 12], :]                # (T,2,2)
            obs = np.any(hips != 0, axis=2)                        # (T,2)
            denom = obs.sum(axis=1)                                # (T,)
            cen = (hips * obs[..., None]).sum(axis=1) / np.maximum(denom, 1)[:, None]
            centres.append(cen)                                    # (T,2)
            valids.append(denom > 0)                               # (T,)
        if mode == "scene":
            cens = np.stack(centres, axis=1)                       # (T,P,2)
            vals = np.stack(valids, axis=1)                        # (T,P)
            n = vals.sum(axis=1)                                   # (T,)
            scene = (cens * vals[..., None]).sum(axis=1) / np.maximum(n, 1)[:, None]
            centres = [scene] * n_persons
            valids = [n > 0] * n_persons
        for p in range(n_persons):
            base = p * JOINTS_PER_PERSON
            block = out[:, base:base + JOINTS_PER_PERSON, :]
            observed = np.any(block != 0, axis=2, keepdims=True)
            ok = observed & valids[p][:, None, None]
            out[:, base:base + JOINTS_PER_PERSON, :] = np.where(
                ok, block - centres[p][:, None, :], block)
        return out

    @staticmethod
    def _scale(xy: np.ndarray) -> np.ndarray:
        m = np.max(np.abs(xy[xy != 0])) if np.any(xy != 0) else 0.0
        return xy / m if m > 0 else xy

    def _load(self) -> None:
        files = sorted(glob.glob(os.path.join(self.data_dir, "**", "*.json"), recursive=True))
        suffix = f"{self.mode}.json"
        for path in files:
            if not os.path.basename(path).endswith(suffix):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    content = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(content, dict) or "data" not in content:
                continue
            label = self._resolve_label(content, path)
            if label is None:
                continue
            for seq_raw in split_sequences_from_raw_data(content["data"]):
                arr = self._process_sequence(seq_raw)
                if arr is not None:
                    self.data_list.append(arr)
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, i: int):
        x = torch.from_numpy(np.ascontiguousarray(self.data_list[i])).float()
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y
