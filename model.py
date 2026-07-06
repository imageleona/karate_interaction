"""
model.py -- Two-person ST-GCN for COCO-17 keypoints.

Implements section 4 of ``YOLO17_WORKSPACE_SPEC.md``:
  * :class:`Graph`     builds a 3-partition (self/centripetal/centrifugal) normalized
    adjacency of shape ``(3, V, V)`` over ``V = num_nodes`` (34 for two-person COCO-17).
  * :class:`GraphConv` spatial graph convolution (1x1 conv -> 3 partitions, learnable
    edge importance, einsum over partitions and neighbours).
  * :class:`STGCNBlock` GraphConv -> temporal Conv2d(9,1) + residual.
  * :class:`STGCN`     input BatchNorm -> 9 ST-GCN blocks -> global pool -> Linear.

The network is fully parameterized by ``num_nodes``; only ``Graph`` holds the keypoint-specific
knobs (``nodes_per_person``, ``center``, ``p1_edges``, wrist indices).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Physical bones for one person, COCO-17 (see spec 4.1).
COCO17_EDGES = [
    (15, 13), (13, 11),          # left leg:  ankle-knee-hip
    (16, 14), (14, 12),          # right leg: ankle-knee-hip
    (11, 12),                    # hips
    (5, 11), (6, 12),            # torso sides (shoulder-hip)
    (5, 6),                      # shoulders
    (5, 7), (7, 9),              # left arm:  shoulder-elbow-wrist
    (6, 8), (8, 10),             # right arm: shoulder-elbow-wrist
    (0, 1), (0, 2),              # nose-eyes
    (1, 3), (2, 4),              # eyes-ears
    (0, 5), (0, 6),              # nose-shoulders (head to torso)
]
WRIST_JOINTS = (9, 10)          # COCO-17 left/right wrist


class Graph:
    """Spatial adjacency over ``num_nodes`` COCO-17 nodes (1 or 2 persons)."""

    def __init__(self, num_nodes: int, strategy: str = "spatial", interaction_mode: str = "full"):
        self.num_nodes = num_nodes
        self.nodes_per_person = 17
        self.center = 11                        # left_hip: closest torso anchor (no pelvis in COCO)
        self.interaction_mode = interaction_mode
        assert num_nodes % self.nodes_per_person == 0, num_nodes
        self.num_persons = num_nodes // self.nodes_per_person
        self.edges = self._build_edges()
        self.A = self._build_adjacency(strategy)

    def _build_edges(self) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        for p in range(self.num_persons):
            off = p * self.nodes_per_person
            edges += [(i + off, j + off) for (i, j) in COCO17_EDGES]

        if self.num_persons == 2 and self.interaction_mode != "none":
            off = self.nodes_per_person
            if self.interaction_mode == "full":
                for i in range(self.nodes_per_person):
                    for j in range(self.nodes_per_person):
                        edges.append((i, j + off))
            elif self.interaction_mode == "hand_cross":
                for w in WRIST_JOINTS:
                    for j in range(self.nodes_per_person):
                        edges.append((w, j + off))          # P1 wrist -> all P2 joints
                        edges.append((w + off, j))          # P2 wrist -> all P1 joints
            else:
                raise ValueError(f"unknown interaction_mode: {self.interaction_mode}")
        return edges

    def _centers(self) -> list[int]:
        return [self.center + p * self.nodes_per_person for p in range(self.num_persons)]

    def _build_adjacency(self, strategy: str) -> torch.Tensor:
        V = self.num_nodes
        adj = np.zeros((V, V), dtype=np.float32)
        for i, j in self.edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

        # BFS from centre node(s) to get hop distance from centre for the partition split.
        dist = np.full(V, np.inf)
        q = deque()
        for c in self._centers():
            dist[c] = 0.0
            q.append(c)
        while q:
            u = q.popleft()
            for v in range(V):
                if adj[u, v] and dist[v] == np.inf:
                    dist[v] = dist[u] + 1
                    q.append(v)

        A = np.zeros((3, V, V), dtype=np.float32)   # 0=self, 1=centripetal, 2=centrifugal
        for i in range(V):
            for j in range(V):
                if i == j:
                    A[0, i, j] = 1.0
                elif adj[i, j]:
                    if dist[j] < dist[i]:
                        A[1, i, j] = 1.0            # neighbour closer to centre
                    else:
                        A[2, i, j] = 1.0            # neighbour farther / equal
        # row-normalize each partition (guard zero rows)
        for k in range(3):
            row = A[k].sum(axis=1, keepdims=True)
            row[row == 0] = 1.0
            A[k] = A[k] / row
        return torch.from_numpy(A)


class GraphConv(nn.Module):
    """Spatial graph convolution: 1x1 conv over channels, one weight set per partition."""

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.num_partitions = A.shape[0]
        self.register_buffer("A", A)
        self.edge_importance = nn.Parameter(torch.ones_like(A))
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_partitions, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        x = self.conv(x)                                        # (N, out*K, T, V)
        x = x.view(n, self.num_partitions, self.out_channels, t, v)
        adj = self.A * self.edge_importance                     # (K, V, V)
        x = torch.einsum("nkctv,kvw->nctw", x, adj)             # sum over partitions and neighbours
        return x.contiguous()


class STGCNBlock(nn.Module):
    """GraphConv (spatial) -> temporal Conv2d(9,1), with residual."""

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor,
                 stride: int = 1, dropout: float = 0.5, residual: bool = True):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1),
                      stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.tcn(self.gcn(x))
        return self.relu(x + res)


# (out_channels, stride) per block; input channels chain from the previous block.
# The 6/4-layer variants only use (in,out,stride) combinations that also exist in the
# 9-layer NTU reference net, so pretrained weights still map layer-by-layer (see
# training.load_pretrained_backbone's shape-based matching).
LAYER_SPECS = {
    9: [(64, 1), (64, 1), (64, 1), (128, 2), (128, 1), (128, 1), (256, 2), (256, 1), (256, 1)],
    6: [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1)],
    4: [(64, 1), (128, 2), (256, 2), (256, 1)],
}


class STGCN(nn.Module):
    """Two-person ST-GCN (9, 6 or 4 blocks). Input ``(N, C, T, V)`` -> logits ``(N, num_classes)``."""

    def __init__(self, num_classes: int, in_channels: int, num_nodes: int,
                 interaction_mode: str = "full", dropout: float = 0.5, num_layers: int = 9):
        super().__init__()
        if num_layers not in LAYER_SPECS:
            raise ValueError(f"num_layers must be one of {sorted(LAYER_SPECS)}, got {num_layers}")
        self.graph = Graph(num_nodes, interaction_mode=interaction_mode)
        A = self.graph.A
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        blocks = []
        c_in = in_channels
        for i, (c_out, stride) in enumerate(LAYER_SPECS[num_layers]):
            blocks.append(STGCNBlock(c_in, c_out, A, stride=stride,
                                     dropout=dropout, residual=(i > 0)))
            c_in = c_out
        self.layers = nn.ModuleList(blocks)
        self.fc = nn.Linear(c_in, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)    # input BN over channel*joint
        x = self.data_bn(x)
        x = x.view(n, c, v, t).permute(0, 1, 3, 2).contiguous()     # back to (N, C, T, V)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=[2, 3])                                       # global avg pool over (T, V)
        return self.fc(x)
