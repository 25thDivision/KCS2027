#!/usr/bin/env python3
"""
Dual-head GNN (message passing) decoder on the detector graph.

KCS original: ../KCS/stim_simulation/models/gnn.py (class GNN).
Changes from the original:
  * two heads (per-qubit + logical) instead of one output layer: the
    original's last Linear becomes head_qubit and a parallel head_logical
    is added on the same features
  * nodes are Stim detectors from model/graph.GraphBuilder (final-Z
    detectors included) instead of (stabilizer, round) nodes; the graph
    is built inside the model, so forward takes only x
  * message passing uses the shared normalized adjacency
    self.adj = D^-1/2 (A+I) D^-1/2 (the original builds a row-normalized
    (A+I)/deg matrix from edge_index on every forward)

Input : (B, 2*num_cycles + 1, H, W) uint8 AUGMENTED syndrome tensor
        (prepare_features). self.node_inputs(x) turns it into node
        features (B, N, IN_FEATS) = [detector bit, 6 static features].
Output: qubit_logits   (B, num_qubits) - per-qubit X-error head
                                         (diagnostics: ECR, parity_LER)
        logical_logits (B, 1)          - logical Z flip head
                                         (official metric: head-LER)

Architecture (h: node states (B, N, .), N = self.num_nodes):
  layers[0]    Linear(IN_FEATS, hidden_dim)
  layers[1..]  num_layers - 1 x Linear(hidden_dim, hidden_dim)
  each layer   h = ReLU(layers[i](self.adj @ h))
  flatten      (B, N * hidden_dim)
  shared       Linear(N * hidden_dim, fc_dim) -> ReLU
  head_qubit   Linear(fc_dim, num_qubits)
  head_logical Linear(fc_dim, 1)

Pure torch only - no torch_geometric or other graph libraries.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.common import (  # noqa: E402,F401
    GraphDecoderBase, IN_FEATS, dual_head_loss)
from model.graph import augment_features  # noqa: E402

NUM_QUBITS = 17


def prepare_features(features, final_bits, code="heavyhex", distance=3):
    """Dataset tensor (B,2C,H,W) + final bits (B,num_qubits) -> augmented input.

    train.py passes the per-qubit flip labels; hardware/run_hw.py passes
    the measured final data bits. Same detector bits either way (see
    model/graph.py)."""
    return augment_features(features, final_bits, code, distance)


class GNNDecoder(GraphDecoderBase):
    """in_channels = 2*num_cycles (same convention as the CNN)."""

    def __init__(self, in_channels=6, num_qubits=NUM_QUBITS,
                 code="heavyhex", distance=3,
                 hidden_dim=64, num_layers=4, fc_dim=128):
        super().__init__(in_channels, code, distance)
        self.layers = nn.ModuleList(
            [nn.Linear(IN_FEATS, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim)
               for _ in range(num_layers - 1)])
        self.shared = nn.Sequential(
            nn.Linear(self.num_nodes * hidden_dim, fc_dim), nn.ReLU())
        self.head_qubit = nn.Linear(fc_dim, num_qubits)
        self.head_logical = nn.Linear(fc_dim, 1)

    def forward(self, x):
        h = self.node_inputs(x)                          # (B, N, IN_FEATS)
        for layer in self.layers:
            h = F.relu(layer(self.adj @ h))
        z = self.shared(h.flatten(1))
        return self.head_qubit(z), self.head_logical(z)


def compute_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                 aux_weight=0.5, qubit_pos_weight=None):
    """Return (total_loss, loss_logical, loss_qubit) - model.common."""
    return dual_head_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                          aux_weight, qubit_pos_weight)
