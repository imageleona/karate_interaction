# Blueprint: Two-Person ST-GCN Action Recognition Workspace (YOLO/COCO-17 keypoints)

This document is a **self-contained specification** for recreating a skeleton-based
action-recognition workspace from scratch in a new folder. It describes the four core
files (`dataset.py`, `model.py`, `training.py`, `test.py`), the exact tensor/JSON shapes
they exchange, and every place that depends on the keypoint layout.

The reference implementation was built for **NTU RGB+D (25 joints/person, 50 nodes)**.
This blueprint targets **YOLO-pose / COCO-17 (17 joints/person, 34 nodes)**. Wherever the
two differ, both values are given so the intent is unambiguous.

---

## 1. What the workspace does

Classifies **two-person interaction** video clips into action classes from per-frame 2D
skeleton keypoints, using a 9-layer **ST-GCN** (spatial-temporal graph convolutional
network). Each person is a graph of joints; the two skeletons are joined by optional
cross-person "interaction" edges.

Data flow:

```
keypoint source ──► training JSON (per class, train/test split) ──► training.py ──► output/<ts>_training/
                                                                          │
                                                              test.py ◄───┘  ──► output/<ts>_test/
```

The **model / dataset / training / test** code is source-agnostic: it only consumes the
training JSON described in §2. How you produce that JSON (running YOLO-pose on videos,
converting existing keypoint dumps, etc.) is a separate ingestion step you write to match
your data.

---

## 2. Data format (the contract every file relies on)

### 2.1 Directory layout

One `train` and one `test` JSON per action class:

```
json_output/
  <CLASS>/
    train/<CLASS>_train.json
    test/<CLASS>_test.json
  ...
```

`<CLASS>` is any label string (e.g. `A050`, or `hug`, `handshake`, …). The loader matches
a file to a class by (a) the `index` field inside the JSON, or (b) the class name appearing
in the filename — so keep the class name in the filename.

### 2.2 JSON file schema

```json
{
  "index": 1,          // 1-based class label; must equal (class position in CLASS_NAMES) + 1
  "data": [            // list of sequences (one per clip/sample in this class+split)
    [                  // sequence  = list of frames (variable length, NOT padded here)
      [ /* frame */ ], // frame     = flat list of floats, fixed length (see §2.3)
      ...
    ],
    ...
  ]
}
```

- `data` is a **ragged** 3-level list: `data[sample][frame][coord]`.
- Sample count and frame count are variable. **Padding/resampling to a fixed frame count
  happens at load time in `dataset.py`, not in the JSON.**
- Every frame row has the **same fixed length** (§2.3).

### 2.3 The flat frame row

Two persons concatenated, each person = all joints, each joint = its coordinates, flattened
as `[x, y, x, y, ...]`:

```
COCO-17, 2 channels (x,y):  frame length = 17 * 2 * 2 = 68 floats
  indices  0–33 : person 1 → [j0_x, j0_y, j1_x, j1_y, ..., j16_x, j16_y]
  indices 34–67 : person 2 → [j0_x, j0_y, j1_x, j1_y, ..., j16_x, j16_y]
```

To read joint `k` (0–16) of person `p` (0 or 1): `row[p*34 + k*2]` = x, `+1` = y.

(NTU-25 reference used length `25*2*2 = 100`; person split at index 50.)

**Optional confidence channel.** If you keep YOLO's per-keypoint confidence as a 3rd
channel, frame length becomes `17 * 3 * 2 = 102` and you must set `INPUT_CHANNELS = 3`
(see §3.1). Default recommendation: **x,y only (2 channels, 68 floats)** — simplest, matches
the model unchanged.

### 2.4 COCO-17 keypoint order (YOLO-pose)

```
0 nose        1 left_eye     2 right_eye    3 left_ear     4 right_ear
5 left_shoulder  6 right_shoulder  7 left_elbow  8 right_elbow
9 left_wrist  10 right_wrist  11 left_hip    12 right_hip
13 left_knee  14 right_knee   15 left_ankle  16 right_ankle
```

### 2.5 Ingestion notes (producing the JSON from YOLO)

- YOLO-pose returns, per frame, an array of detected people, each `(17, 2)` xy (plus conf).
- For interaction you need **exactly two people per frame in a consistent order.** Pick the
  two highest-confidence / largest-box detections and order them deterministically (e.g. by
  bounding-box x-center: left person = P1). Ideally use a tracker to keep P1/P2 stable
  across frames. Frames missing a second person can be zero-filled or dropped.
- Normalize coordinates however you like (raw pixels work; the model applies input
  BatchNorm). Missing joints → `0.0`.
- Apply an 80/20 per-class train/test split with a fixed seed for reproducibility.

---

## 3. `dataset.py` — PyTorch Dataset

### 3.1 Module constants

```python
INPUT_CHANNELS   = 2          # 2 = (x,y); set 3 if frame rows include confidence
DEFAULT_NUM_FRAMES = 150      # fixed time length after pad/resample
JOINTS_PER_PERSON  = 17       # COCO-17  (NTU used 25)
CLASS_NAMES = [ ... ]         # ordered list of class labels; index in this list + 1 == JSON "index"
```

### 3.2 Responsibilities

A `torch.utils.data.Dataset` subclass (call it `SkeletonDataset`) that, given a root dir,
a `class_names` list, and `mode ∈ {"train","test"}`:

1. **Recursively globs** `**/*.json` under the data dir; keeps files whose name ends with
   `<mode>.json`.
2. For each file: reads `content`; resolves the label from `int(content["index"]) - 1`
   (fallback: first class name found in the filename). Skip if unresolved.
3. Splits `content["data"]` into sequences (handle both a single sequence and a list of
   sequences — see helper below).
4. For each sequence:
   - `np.array(seq, dtype=float32)`; must be 2-D `(frames, feat_dim)`, else skip.
   - `np.nan_to_num(...)` to kill NaN/Inf.
   - **(optional) normalize** — centering/scaling (see §3.4).
   - **reshape/split into two persons** → `(frames, 34, 2)` (helper below).
   - **pad or resample** the time axis to `num_frames`.
   - **transpose** to channel-first: final per-sample array shape **`(C, T, V)` = `(2, 150, 34)`**.
   - append array + label.

`__getitem__` returns `(torch.FloatTensor of shape (C,T,V), label:long)`.

### 3.3 Key helpers

**Sequence splitter** — a raw `data` field may be a single sequence or a list of sequences:

```python
def split_sequences_from_raw_data(raw):
    if not isinstance(raw, list) or not raw: return []
    # list-of-sequences if raw[0][0] is itself a list (i.e. 3 levels deep)
    if isinstance(raw[0], list) and raw[0] and isinstance(raw[0][0], list):
        return raw
    return [raw]
```

**Two-person extractor** — flat frame → `(frames, num_joints, 2)`:

```python
def extract_two_person_xy_flat(seq):           # seq: (frames, feat_dim)
    frames, feat_dim = seq.shape
    if feat_dim % 2: return None
    detected = feat_dim // INPUT_CHANNELS
    tmp = seq.reshape(frames, detected, INPUT_CHANNELS)
    if detected == 34:                          # 17 joints x 2 persons
        return np.concatenate([tmp[:, 0:17], tmp[:, 17:34]], axis=1)
    if detected == 17:                          # single person
        return tmp
    return None
```

**Time pad/resample** — pad short clips with zero frames, linspace-subsample long ones:

```python
def pad_resample_time(arr, num_frames):         # arr: (T, ...)
    t = arr.shape[0]
    if t < num_frames:
        pad = np.zeros((num_frames - t,) + arr.shape[1:], np.float32)
        return np.vstack((arr.astype(np.float32), pad))
    idx = np.linspace(0, t - 1, num_frames).astype(int)
    return arr[idx].astype(np.float32)
```

### 3.4 Normalization (optional, off by default)

Constructor flags `do_center` / `do_scale`:
- `do_center`: subtract a root joint's xy from every joint each frame. NTU used joint 0
  (base of spine). **COCO joint 0 is the nose — a poor center.** If you enable centering,
  use the **hip midpoint** (mean of joints 11 and 12) instead.
- `do_scale`: divide by the max absolute coordinate (ignoring zeros) to unit-scale.

Leave both **False** unless you have a reason; the model's input BatchNorm handles scale.

### 3.5 Self-configuring channels

After loading, set `self.in_channels = data_list[0].shape[0]` so training can read the
actual channel count (2 or 3) back off the dataset.

---

## 4. `model.py` — ST-GCN

Two classes matter: **`Graph`** (builds the adjacency) and **`STGCN`** (the network).
`GraphConv` and `STGCNBlock` are internal building blocks.

### 4.1 `Graph` — spatial adjacency over 34 nodes

Builds a **3-partition** normalized adjacency tensor `A` of shape `(3, V, V)` where
`V = num_nodes` (34 for two-person COCO-17). Partitions = **self / centripetal / centrifugal**
(the standard ST-GCN "spatial" partitioning).

Constructor: `Graph(num_nodes, strategy="spatial", interaction_mode="full")`.

Fields to set for COCO-17:
```python
self.nodes_per_person = 17
self.center = 11          # BFS root for centripetal/centrifugal split.
                          # No pelvis exists in COCO-17; left_hip (11) is the
                          # closest torso anchor. Only affects the partition split.
```

**Physical bones (one person), COCO-17:**
```python
p1_edges = [
    (15,13),(13,11),            # left leg:  ankle-knee-hip
    (16,14),(14,12),            # right leg: ankle-knee-hip
    (11,12),                    # hips
    (5,11),(6,12),              # torso sides (shoulder-hip)
    (5,6),                      # shoulders
    (5,7),(7,9),                # left arm:  shoulder-elbow-wrist
    (6,8),(8,10),               # right arm: shoulder-elbow-wrist
    (0,1),(0,2),                # nose-eyes
    (1,3),(2,4),                # eyes-ears
    (0,5),(0,6),                # nose-shoulders (head to torso)
]
```

**Two persons:** duplicate the edges with `+17` offset for person 2, then add
cross-person edges according to `interaction_mode`:
- `"none"`   — no cross edges (two independent skeletons).
- `"full"`   — every P1 joint ↔ every P2 joint (17×17 = 289 cross edges each direction).
- `"hand_cross"` — only wrists (**COCO 9, 10**) linked to all joints on the other person,
  both directions. (NTU used wrists 6, 10.)

**Adjacency construction algorithm:**
1. Build symmetric binary `adj (V,V)` from the edge list.
2. BFS from the center node(s) — `[center]` for one person, `[center, center+17]` for two —
   to get `dist_from_center[v]` for every node.
3. For each ordered pair `(i,j)`: partition 0 if `i==j` (self); else if adjacent, partition 1
   when `dist[j] < dist[i]` (centripetal, toward center) else partition 2 (centrifugal).
4. **Row-normalize** each partition (divide each row by its sum; guard sum==0 → 1).
5. Return as `torch.FloatTensor` of shape `(3, V, V)`.

### 4.2 `GraphConv`

```
in:  x (N, C_in, T, V)
1x1 conv C_in -> C_out * 3      # one weight set per partition
reshape to (N, 3, C_out, T, V)
adj = A * edge_importance        # edge_importance: learnable Parameter, same shape as A, init ones
out = einsum('nkctv,kvw->nctw', x, adj)   # sum over partitions k and neighbors v
```

### 4.3 `STGCNBlock`

`gcn` (GraphConv) → `tcn`, plus a residual:
```
tcn = Sequential(BatchNorm2d, ReLU, Conv2d(C,C,kernel=(9,1),stride=(stride,1),pad=(4,0)),
                 BatchNorm2d, Dropout(dropout))
residual = Identity() if (in==out and stride==1)
           else Sequential(Conv2d(in,out,1,stride=(stride,1)), BatchNorm2d)
forward: relu(tcn(gcn(x)) + residual(x))
```
The `(9,1)` temporal conv is the temporal graph convolution; `stride` downsamples time.

### 4.4 `STGCN` (the network)

Constructor: `STGCN(num_classes, in_channels, num_nodes, interaction_mode="full", dropout=0.5)`.

```
graph = Graph(num_nodes, interaction_mode=interaction_mode); A = graph.A   # (3,V,V)
data_bn = BatchNorm1d(in_channels * num_nodes)      # input normalization

9 blocks (channels; stride):
  layer1  in_channels->64
  layer2  64->64
  layer3  64->64
  layer4  64->128   stride=2      # temporal downsample #1
  layer5  128->128
  layer6  128->128
  layer7  128->256  stride=2      # temporal downsample #2
  layer8  256->256
  layer9  256->256
fc = Linear(256, num_classes)
```

**Forward** (`x : (N, C, T, V)` → logits `(N, num_classes)`):
```
# input BN over channel*joint:
x = x.permute(0,1,3,2).reshape(N, C*V, T); x = data_bn(x)
x = x.reshape(N, C, V, T).permute(0,1,3,2)      # back to (N,C,T,V)
x = layer1..layer9(x)
x = x.mean(dim=[2,3])                            # global average pool over (T,V) -> (N,256)
return fc(x)
```

The model is **completely parameterized by `num_nodes`** — nothing else changes between
NTU-25 (`num_nodes=50`) and COCO-17 (`num_nodes=34`). Only `Graph`'s `nodes_per_person`,
`center`, `p1_edges`, and the wrist indices are keypoint-specific.

---

## 5. `training.py` — training loop

CLI (argparse) with sensible defaults:

| Arg | Default | Meaning |
|---|---|---|
| `--data-dir` | `./json_output` | root with `*train.json` |
| `--epochs` | 200 | |
| `--batch-size` | 64 | |
| `--lr` | 0.001 | Adam LR |
| `--val-ratio` | 0.2 | split off val from train |
| `--seed` | 42 | |
| `--num-frames` | `None`→150 | pad/resample target `T` |
| `--no-interaction` | flag | sets `interaction_mode="none"` (else `"full"`) |

Flow:
1. `CLASSES = CLASS_NAMES` (from dataset module). `interaction_mode = "none" if --no-interaction else "full"`.
2. `device = cuda if available else cpu`. Make `output/<timestamp>_training/`.
3. `full = SkeletonDataset(data_dir, class_names=CLASSES, mode="train", num_frames=...)`.
   Read `in_ch = full.in_channels`.
4. `random_split` into train/val by `val_ratio` with a seeded generator. Two `DataLoader`s
   (train shuffled, val not).
5. Build model: **`num_nodes = 34`** (COCO-17 two-person):
   ```python
   model = STGCN(num_classes=len(CLASSES), in_channels=in_ch,
                 num_nodes=34, interaction_mode=interaction_mode).to(device)
   ```
6. `optimizer = Adam(lr, weight_decay=5e-4)`; `criterion = CrossEntropyLoss()`;
   `scheduler = MultiStepLR(milestones=[0.5*epochs, 0.75*epochs], gamma=0.1)`.
7. **Save `config.json`** into the run dir — this is what `test.py` reads back:
   ```json
   {"in_channels": <int>, "num_nodes": 34, "interaction_mode": "<mode>",
    "classes": [...], "num_frames": <int>}
   ```
8. Epoch loop: standard train step (zero_grad → forward → CE loss → backward → step),
   then eval on val (no_grad). Track train/val loss+acc. Append a row to `epoch_log.csv`.
   Save `best_model.pth` whenever val accuracy improves. `scheduler.step()` each epoch.
9. After training, save `loss_graph.png` and `accuracy_graph.png` (matplotlib).

(The reference also logs to Weights & Biases behind a `--no-wandb` flag; **optional** — omit
unless you want it, and if included, guard all wandb calls so `--no-wandb` fully disables it.)

**The only line that changes vs NTU is `num_nodes=34`** (and it must also be `34` in the
saved `config.json`).

---

## 6. `test.py` — evaluation

1. `--output-dir` (a `..._training` run folder). If omitted, **auto-pick the newest
   `output/*_training/` that contains both `config.json` and `best_model.pth`.**
2. `--data-dir` (root with `*test.json`).
3. Load `config.json`. Build the test dataset with **`class_names = cfg["classes"]`,
   `mode="test"`, `num_frames=cfg["num_frames"]`**.
4. Rebuild the model straight from config — this is why config is saved:
   ```python
   model = STGCN(num_classes=len(cfg["classes"]), in_channels=cfg["in_channels"],
                 num_nodes=cfg["num_nodes"], interaction_mode=cfg["interaction_mode"]).to(device)
   model.load_state_dict(torch.load(best_model.pth, map_location=device, weights_only=True))
   model.eval()
   ```
   Because the model is config-driven, **`test.py` needs no keypoint-count edits** — it works
   for 34 or 50 as long as the training `config.json` says so.
5. Run inference over the test loader (no_grad), collect preds + labels.
6. Emit into a fresh `output/<timestamp>_test/`:
   - `test_report.txt` — sklearn `classification_report` (per-class precision/recall/F1).
   - `test_cm.png` + `test_cm.txt` — row-normalized confusion matrix (heatmap + text).
   - `checkpoint_source.txt` — the training dir that was used.

---

## 7. Reproduction checklist (NTU-25 ➜ YOLO/COCO-17)

Everything below is the complete set of keypoint-dependent knobs. Get these right and the
rest of the code is identical.

| Location | NTU-25 | **COCO-17** |
|---|---|---|
| `dataset.py` `JOINTS_PER_PERSON` | 25 | **17** |
| `dataset.py` `num_joints` / node count | 50 | **34** |
| `dataset.py` extractor person split | `0:25 / 25:50` (feat 100) | **`0:17 / 17:34` (feat 68)** |
| `dataset.py` `INPUT_CHANNELS` | 2 | **2** (or 3 if using confidence) |
| `model.py` `Graph.nodes_per_person` | 25 | **17** |
| `model.py` `Graph.center` | 0 (spine base) | **11 (left_hip; no pelvis in COCO)** |
| `model.py` `Graph.p1_edges` | NTU-25 bones | **COCO-17 bones (§4.1)** |
| `model.py` wrist indices (`hand_cross`) | (6, 10) | **(9, 10)** |
| `training.py` `num_nodes=` and `config.json` | 50 | **34** |
| `test.py` | — | no change (reads `num_nodes` from config) |
| JSON frame length | 100 | **68** (2ch) / 102 (3ch) |
| `CLASS_NAMES` | 26 NTU actions | your action set |

**Build order for a fresh workspace:**
1. Write `dataset.py`, `model.py`, `training.py`, `test.py` per §3–6 with the COCO-17 column.
2. Write an ingestion script that turns your YOLO-pose output into the §2 JSON layout
   (68-float frames, per-class train/test split, `index` = class position + 1).
3. Smoke test: build `STGCN(num_classes=N, in_channels=2, num_nodes=34)`, feed a random
   `(1, 2, 150, 34)` tensor, confirm it returns `(1, N)` logits.
4. `python training.py` → check a `config.json` with `num_nodes: 34` and a `best_model.pth`
   appear under `output/<ts>_training/`.
5. `python test.py` → confirm it auto-loads that run and writes a report + confusion matrix.

### 7.1 Sanity checks
- Every JSON frame row length is constant and equals `68` (2ch COCO-17).
- `index` in each JSON == (its class position in `CLASS_NAMES`) + 1.
- A loaded sample tensor is `(2, 150, 34)` = `(channels, frames, nodes)`.
- The adjacency graph is **connected per person** (BFS from `center` reaches all 17 joints);
  otherwise `dist_from_center` stays `inf` and the partition split misbehaves.
