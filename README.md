# Two-Person ST-GCN — Karate Technique & Point Recognition

Skeleton-based recognition of karate techniques and point decisions from two-person
interaction clips, using a spatial-temporal graph convolutional network (ST-GCN) with
cross-person interaction edges and transfer learning from NTU RGB+D.

This README documents the complete method — data, preprocessing, splits, augmentation,
normalization, architecture, training protocol, experimental design and evaluation — in
enough detail to write the Methods section of the accompanying research.

---

## 1. Data

### 1.1 Source material
- 537 clips cut from **28 source videos** (`split_report.csv` maps every clip to its source),
  each video containing sparring exchanges of exactly one technique.
- Per clip: 2D poses of the **two athletes**, estimated with YOLO-pose (COCO-17 keypoints),
  at a fixed **120 frames** per clip, in **raw pixel coordinates** (≈1920×1080 frames).

### 1.2 Skeleton format
Each JSON file is `{"index": <1-based class label>, "data": [sequence, ...], "clip_ids":
[<clip_id per sequence>]}`; a sequence is 120 frames; a frame is a flat list of **68
floats** = 2 persons × 17 COCO joints × (x, y). `clip_ids[k]` names `data[k]`
(e.g. `00kizami-10_01`; augmented copies are `<clip_id>#augN`) and is the join key for all
per-clip metadata (`split_report.csv`, `camera_angle.csv`, test predictions):

```
indices  0–33 : person 1  [j0_x, j0_y, ..., j16_x, j16_y]
indices 34–67 : person 2  (same layout)
```

COCO-17 joint order: nose, L/R eye, L/R ear, L/R shoulder, L/R elbow, L/R wrist, L/R hip,
L/R knee, L/R ankle. Missing keypoints are stored as `0.0` (or `null`; both are treated as
missing and zero-filled at load). Person 1 is the left athlete.

### 1.3 Class labelings (two parallel tasks on the same clips)
- **4-class (technique):** kizami (jab punch), chudan (mid punch), chudan-geri (mid kick),
  jodan-geri (high kick). Train-pool/test clips: 112/35, 80/41, 84/40, 98/47.
- **8-class (technique × point):** each technique split by whether the referee awarded a
  point (`*_point` / `*_no_point`). Chance levels: 25% (4-class), 12.5% (8-class).
- **15-class (technique × point-or-reason):** the no-point clips further split by WHY no
  point was scored (`too_far`, `blocked`, `bad_aim` (chudan/chudan-geri only),
  `not_extended` (kizami only)) — manually annotated in the extraction workspace
  (`clips_detailed/`, exported by `make_reason_labels.py`); the reason is also the
  `reason` column of `split_report.csv`. Only the populated technique×reason combinations
  exist, hence 15 classes, with thin cells (9–33 train clips before augmentation).

### 1.4 Train / validation / test split — source-grouped (no footage leakage)
- **Test (163 clips):** fixed, held-out set; its 6+ source videos never appear in training
  (grouping recorded in `split_report.csv`).
- **Validation (106 clips):** produced by holding out **whole source videos** from the train
  pool (~20% of clips per technique; seeded, frozen on disk as `<class>_val.json`). Held-out
  sources: `00kizami-8/-10/-12`, `01chudan-6`, `02chudankeri-6`, `03jodankeri-1`
  (recorded in `skeleton_dataset_augmented/val_split.json`). Because one source = one
  technique, the same assignment is consistent across both labelings.
- **Train (268 original clips → 1340 after augmentation).**
- Rationale: clips from one video share athletes, camera and lighting; a random per-clip
  split lets the model validate on footage it effectively saw. With source grouping the
  validation accuracy tracks test accuracy (before this fix: val ≈ 0.99 vs test ≈ 0.42).
- The split is performed **before** augmentation, so no augmented copy of a validation clip
  can appear in training. Verified: 0 duplicated clips, 0 shared sources between train/val.

## 2. Augmentation (`augment_skeleton_train.py`)

Applied **only to the training portion**; validation and test stay clean. Each original
sequence yields 4 augmented copies (**5× training data**). Geometric parameters are sampled
once per sequence and applied to every frame. Defaults (used for all experiments; seed 0):

| Transform | Setting |
|---|---|
| Horizontal flip | p = 0.5; `x' = W − x` (W = 1920) + COCO left/right joint swap + person 1↔2 swap (preserves "left athlete = person 1") |
| Per-keypoint Gaussian jitter | σ = 6 px per axis, displacement radius capped at 15 px, clipped to the frame |
| Keypoint dropout | p = 0.02 per observed keypoint per frame (set to 0,0) |
| Scale / rotation / translation | **disabled** (0) — they perturb the vertical-height cue separating mid from high kicks and hurt test accuracy |
| Time reversal | disabled (a reversed strike is not a valid technique) |

Missing keypoints are preserved as missing through all transforms.

## 3. Coordinate normalization (`--center`, `--scale`)

Raw pixel coordinates encode absolute image position, which does not transfer across
recording setups (diagnosed by a systematic chudan-geri↔jodan-geri swap on test and a large
val/test gap). Two centering modes, applied per frame at load time in `dataset.py`:

- **`--center scene`** (used in the final experiments): both skeletons are translated by the
  mean of the two per-person **hip midpoints** (mean of joints 11, 12; only observed hips are
  averaged). Removes camera translation while **preserving the relative position between the
  two athletes** — required by the cross-person interaction edges.
- **`--center person`**: each skeleton translated by its own hip midpoint (strongest encoding
  of own-body-relative pose, but discards inter-person geometry). Available, not used in the
  final runs.
- **`--scale`**: after centering, the whole sequence is divided by its maximum absolute
  coordinate (removes camera-zoom / body-size scale).

Missing keypoints stay 0. The settings are stored in each run's `config.json` and re-applied
automatically at test time.

## 4. Model (`model.py`)

### 4.1 Graph over 34 nodes (2 × COCO-17)
- **Within-person bones:** 18 anatomical edges per person (limbs, torso, head; the standard
  COCO skeleton plus nose–shoulder links), duplicated with +17 offset for person 2.
- **Cross-person interaction edges** (`interaction_mode`):
  - `full`: every P1 joint ↔ every P2 joint (17 × 17 = 289 joint pairs);
  - `hand_cross`: only wrists (joints 9, 10) ↔ all joints of the other person (68 pairs);
  - `none`: two disconnected skeletons.
- **Spatial partitioning** (Yan et al.'s ST-GCN "spatial" strategy): the adjacency is split
  into 3 partitions — self, centripetal, centrifugal — by BFS hop distance from a per-person
  center node. COCO-17 has no pelvis joint, so the **left hip (joint 11)** is used as the BFS
  root. Each partition is **row-normalized** (each row divided by its degree in that
  partition), giving A ∈ R^(3×34×34).

### 4.2 Network
Input `(N, C=2, T=120, V=34)` → BatchNorm over C·V → **9 ST-GCN blocks** → global average
pool over (T, V) → linear classifier. Channels: 64, 64, 64, 128 (temporal stride 2), 128,
128, 256 (stride 2), 256, 256; ≈ **3.05 M parameters**. Each block:
- **Spatial graph convolution:** 1×1 conv producing one weight set per partition, then
  `einsum('nkctv,kvw->nctw', x, A_k ∘ M_k)` where **M_k (edge importance)** is a learnable
  per-edge, per-partition multiplier initialized to 1 — one M per block.
- **Temporal convolution:** 9×1 conv along time, BN, dropout 0.5, residual connection.

Reduced-depth variants (6 / 4 blocks, 2.00 M / 1.75 M params) exist behind `--num-layers`
but were not used in the final experiments (they degrade with skip-mapped pretraining).

## 5. Transfer learning from NTU RGB+D

- Source checkpoints: two-person ST-GCN trained on 26 NTU RGB+D interaction classes
  (25 joints/person, 50 nodes) in `../nturgb_interaction/output/`.
- The conv/BN backbone is independent of the node count, so **141 tensors = 3.02 M
  parameters (99% of the model) transfer** directly from 50-node NTU to 34-node COCO-17.
  Kept freshly initialized (dimensionally impossible to transfer): the input BatchNorm
  (C·V differs), the adjacency and all `edge_importance` tensors ((3,50,50) vs (3,34,34)),
  and the classifier head.
- **Matched initialization:** interaction=`full` models start from an NTU checkpoint trained
  *with* interaction edges (`del03_interaction`); interaction=`none` models from one trained
  *without* (`del03_no_interaction`) — so the init × interaction comparison is unconfounded.

## 6. Training protocol (`training.py`)

| Setting | Value |
|---|---|
| Optimizer | Adam, lr **3e-4** (selected by a prior sweep over {1e-4, 2e-4, 3e-4}) |
| Weight decay | 5e-4, **excluded for `edge_importance`** (decay uniformly shrinks all edge weights ~6%, drowning the differential signal) |
| LR schedule | ×0.1 at 50% and 75% of epochs (epochs 20, 30) |
| Loss | Cross-entropy with **label smoothing 0.1** (selected by a prior regularization comparison; removes the val-loss blowup and improved test accuracy) |
| Dropout | 0.5 in every block |
| Epochs / batch | 40 / 64 |
| Model selection | checkpoint with the **best validation accuracy** (`best_model.pth`) |
| Data | augmented train (§2), clean source-grouped val (§1.4); identical files for every run |

Each run writes `output/<timestamp>_<tag>_training/` containing `config.json` (every setting
above, including normalization and pretrained path — `test.py` rebuilds model and
preprocessing from it), `best_model.pth`, `epoch_log.csv`, loss/accuracy curves, and the
edge-importance visualization. Runs are also logged to Weights & Biases.

## 7. Experimental design

**Final comparison** (`run_final_comparison.bat`): a factorial over the method components,
everything else held constant at the settings of §6:

- **Datasets:** 4-class, 8-class
- **Interaction:** `full` vs `none`
- **Initialization:** pretrained (matched NTU checkpoint) vs from scratch
- **Input variant:** raw pixels vs normalized (`--center scene --scale`)
- **Seeds:** 42, 43, 44 (3 repetitions; seeds affect the fresh-parameter init, batch
  shuffling and dropout — the data and splits are identical for every run)

= 2 × 2 × 2 × 3 = 24 runs per input variant, **48 final runs** total. Each run is
train → test → edge-visualization; a failed/silently-crashed training is detected (the run
pointer file is only written on success) and retried once.

Preliminary experiments (same pipeline, before the final sweep): a 6-pattern × 3-LR sweep
(18 runs) to fix the learning rate, and a 5-config regularization comparison × 2 datasets
(10 runs) to fix label smoothing / dropout / depth.

## 8. Evaluation

- **Test set:** the 163 held-out clips (§1.4), never augmented, sources unseen in training.
- **Reported metric:** overall test accuracy as **mean ± std over the 3 seeds** per cell.
  The std is the **population standard deviation (ddof = 0)** of the three per-seed
  accuracies (`summarize_results.py`, written to `output/final_comparison_summary.csv`).
- Per run, `test.py` also writes a per-class precision/recall/F1 report
  (sklearn `classification_report`) and a **row-normalized confusion matrix**
  (rows = true class; each row sums to 1) as PNG + text.
- Because n = 3, individual cell differences are supported primarily by **sign consistency
  across pairings** (each factor is positive in all 4 of its pairings, in both variants)
  rather than per-cell significance tests.

### Final results (test accuracy, mean ± std over 3 seeds)

| dataset | interaction | init | raw pixels | normalized |
|---|---|---|---|---|
| 4-class | none | scratch | 0.227 ± 0.091 | 0.395 ± 0.138 |
| 4-class | none | pretrained | 0.344 ± 0.005 | 0.663 ± 0.087 |
| 4-class | full | scratch | 0.323 ± 0.064 | 0.593 ± 0.015 |
| 4-class | full | pretrained | 0.425 ± 0.073 | **0.757 ± 0.021** |
| 8-class | none | scratch | 0.125 ± 0.045 | 0.372 ± 0.075 |
| 8-class | none | pretrained | 0.178 ± 0.067 | 0.536 ± 0.028 |
| 8-class | full | scratch | 0.182 ± 0.028 | 0.513 ± 0.084 |
| 8-class | full | pretrained | 0.256 ± 0.019 | **0.632 ± 0.026** |

All three components are positive in every pairing: normalization +17…+38 pp, pretraining
+5…+27 pp, interaction +6…+20 pp; effects are approximately additive.

## 9. Learned edge-importance analysis (`visualize_edges.py`, `make_paper_figure.py`)

What is visualized is the **learned edge multiplier**: for each joint pair (i, j),
`m(i,j) = Σ_k (A_k ∘ M_k)(i,j) / Σ_k A_k(i,j)` — the adjacency-weighted mean of the learned
edge-importance over the 3 partitions. Because M initializes to 1, m is exactly 1.0 before
training; deviations (shown red > 1 / blue < 1 on a diverging colormap) are purely what
training changed. (Plotting the raw product A ∘ M instead is misleading: the row
normalization makes BFS-center rows ~8× larger *by construction*, before any learning.)

- `visualize_edges.py` (run automatically after each training): per layer, the two skeletons
  with bones colored by within-person multipliers + a 17×17 cross-person heatmap
  (P1 joints × P2 joints, both edge directions averaged).
- `make_paper_figure.py`: condensed paper figure — cross-person multipliers **averaged over
  the 9 layers**, 4-class and 8-class side by side (PNG + editable SVG/PDF in `output/`).

Findings: the 8-class (point-judgment) model strengthens **wrist↔wrist / wrist↔elbow** edges
most (strike-versus-guard), with the mirror sidedness of facing athletes (P1-right ↔
P2-left); both models suppress hip-related edges — consistent with hip-centered coordinates
carrying no information and counteracting the construction bias of the BFS-center partition.

## 10. Data-quality diagnostics (limitations)

Measured from the skeletons (`0 = hip level, 1 = head level`, ankle apex relative to own hip
normalized by hip–nose distance):
- Leg keypoints are essentially never *missing* in kick clips (≈0%), but their height is
  **mis-measured on scoring kicks**: point-awarded jodan-geri register a median apex of 0.77
  (train) / 0.32 (test) instead of ≈1.0, while missed kicks measure correctly (0.73–0.80).
  Interpretation: at contact the kicking leg occludes/overlaps the opponent, which is where
  pose estimation fails — i.e. the sensor degrades precisely at the decisive moment.
- A single threshold on this one feature separates the two kicks at 82% (train) / 72% (test),
  which upper-bounds kick-pair recognition from these poses; the residual model errors
  concentrate exactly there (chudan-geri, especially `chudankeri_point`).

## 11. Repository guide

| File | Purpose |
|---|---|
| `augment_skeleton_train.py` | source-grouped train/val split + augmentation (§1.4, §2) → `skeleton_dataset_augmented/`; carries per-sequence `clip_ids` through (augmented copies are `<clip_id>#augN`) |
| `dataset.py` | `SkeletonDataset`: JSON → `(2,120,34)` tensors; centering/scaling (§3); optional per-clip camera-angle feature `[sin θ, cos θ]` via `camera_angle_csv` (§13) |
| `model.py` | `Graph` (§4.1), `STGCN` (§4.2); `extra_feature_dim` concatenates per-clip features to the pooled embedding before the classifier |
| `training.py` | training protocol (§6); `--pretrained` transfer (§5); `--camera-angle-csv` (§13) |
| `test.py` | held-out evaluation; rebuilds model + preprocessing from `config.json`; writes `per_clip_predictions.csv` (clip-level results, join key for §13) |
| `analyze_camera_angle.py` | test accuracy vs camera viewing angle from `per_clip_predictions.csv` + `camera_angle.csv` (§13) |
| `visualize_edges.py` | per-layer learned-multiplier visualization (§9) |
| `make_paper_figure.py` | condensed paper figure (§9) |
| `summarize_results.py` | aggregates runs → mean ± std table + CSV (§8) |
| `run_final_comparison.bat` | the 24-run factorial; `norm` keyword = normalized variant (§7) |
| `run_lr_sweep.bat`, `run_regularization_compare.bat`, `run_experiments_with_test.bat`, `run_all_experiments.bat`, `run_all_interactions.bat` | preliminary sweeps |
| `skeleton_dataset/` | original clips + `split_report.csv` (clip → source/split manifest) |
| `skeleton_dataset_augmented/` | generated training tree + `val_split.json` (held-out sources) |
| `YOLO17_WORKSPACE_SPEC.md` | original blueprint this workspace was built from |

## 12. Reproducing

```bash
conda activate limu_aug                       # numpy-only env (never conda base)
python augment_skeleton_train.py --seed 0     # split + augment -> skeleton_dataset_augmented/

conda activate GNN                            # torch 2.5 + CUDA, sklearn, matplotlib, wandb
run_final_comparison.bat 40                   # 24 raw runs  (train -> test -> visualize each)
run_final_comparison.bat 40 norm              # 24 normalized runs
python summarize_results.py                   # mean +/- std table -> output/final_comparison_summary.csv
python make_paper_figure.py                   # paper figure -> output/edge_importance_paper.{png,svg,pdf}
```

Single runs: `python training.py --data-dir skeleton_dataset_augmented/technique_4class
--interaction-mode full --lr 3e-4 --label-smoothing 0.1 --center scene --scale
--pretrained ../nturgb_interaction/output/20260626_012627_del03_interaction`
then `python test.py --data-dir skeleton_dataset/technique_4class`.

Dependencies: `requirements.txt` (torch ≥ 2.0, numpy, matplotlib, scikit-learn; wandb
optional — disable with `--no-wandb`).

## 13. Camera angle (mat homography)

Per-clip camera viewing angle relative to the two athletes, **folded to [0°, 90°]**:
90° = camera perpendicular to the athlete axis (best visibility), 0° = in line with it
(one athlete occludes the other). Pipeline (in the extraction workspace
`..\20260701`, one manual annotation per source video — the camera is static per video):

1. `annotate_court.py` — click the named mat reference points of `court_geometry.json`
   (court corners + the two red start rectangles; WKF 8 m × 8 m defaults, editable) on one
   frame per `normalized\*_norm.mp4`. Fits the image→mat homography (≥4 points), writes
   `court_points.json` + a reprojected-grid check image per video (`court_annotation\`).
2. `camera_angle.py` — recovers the camera ground position from each homography (focal
   from the two orthogonality constraints of `H = K[r1 r2 t]`, principal point at image
   center; both constraints solved jointly in least squares since one is degenerate when a
   mat axis is image-parallel), maps each athlete's observed-ankle mean (hip fallback)
   through H per frame, and writes `camera_angle.csv` (per clip: median/min/max angle, mat
   positions, camera position/height/focal) + top-down QA plots. Synthetic-camera tests:
   ≤0.1 m camera-position and ≤0.4° angle error at 1 px annotation noise.

Uses in this repo (join key = `clip_ids`, §1.2):
- **Model input:** `training.py --camera-angle-csv <path>` feeds `[sin θ, cos θ]` of each
  clip's median angle into the classifier head (`STGCN(extra_feature_dim=2)`; the
  pretrained backbone is untouched, augmented copies inherit their original's angle —
  the horizontal flip leaves the folded angle unchanged; missing angle → zeros). Stored in
  `config.json`; `test.py` re-applies it automatically.
- **Analysis:** `test.py` writes `per_clip_predictions.csv` per test run;
  `analyze_camera_angle.py` joins these with `camera_angle.csv` → accuracy per angle bin
  (overall + per class) as CSV + figure, to quantify the occlusion effect (§10).
