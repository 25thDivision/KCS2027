#!/usr/bin/env python3
"""
Dual-head Graph Mamba decoder on the detector graph.

KCS original: ../KCS/stim_simulation/models/graph_mamba.py (class GraphMamba).
Changes from the original:
  * two heads (per-qubit + logical) instead of one output layer: the
    original's last Linear becomes head_qubit and a parallel head_logical
    is added on the same features
  * nodes are Stim detectors from model/graph.GraphBuilder (final-Z
    detectors included) instead of (stabilizer, round) nodes; the graph
    is built inside the model, so forward takes only x
  * the degree ordering is a buffer computed once with a stable sort
    (self.order / self.inverse_order); the original recomputes it with
    torch_geometric.utils.degree on every forward

Input : (B, 2*num_cycles + 1, H, W) uint8 AUGMENTED syndrome tensor
        (prepare_features). self.node_inputs(x) turns it into node
        features (B, N, IN_FEATS) = [detector bit, 6 static features].
Output: qubit_logits   (B, num_qubits) - per-qubit X-error head
                                         (diagnostics: ECR, parity_LER)
        logical_logits (B, 1)          - logical Z flip head
                                         (official metric: head-LER)

Architecture (h: node states (B, N, .), N = self.num_nodes):
  reorder      h = h[:, self.order]: nodes sorted by degree, highest
               first (provided buffer)
  embedding    Linear(IN_FEATS, d_model) -> LayerNorm(d_model)
  layers       num_layers x Mamba(d_model=d_model, d_state=d_state,
               d_conv=d_conv, expand=expand) with its own LayerNorm:
                 h = LayerNorm(h + Dropout(mamba(h)))
  restore      h = h[:, self.inverse_order]
  flatten      (B, N * d_model)
  shared       Linear(N * d_model, fc_dim) -> ReLU -> Dropout(dropout)
  head_qubit   Linear(fc_dim, num_qubits)
  head_logical Linear(fc_dim, 1)
  mamba_ssm needs CUDA; without the package the constructor raises
  ImportError.

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

try:
    from mamba_ssm import Mamba
except ImportError:          # reported when the model is constructed
    Mamba = None

NUM_QUBITS = 17


def prepare_features(features, final_bits, code="heavyhex", distance=3):
    """Dataset tensor (B,2C,H,W) + final bits (B,num_qubits) -> augmented input.

    train.py passes the per-qubit flip labels; hardware/run_hw.py passes
    the measured final data bits. Same detector bits either way (see
    model/graph.py)."""
    return augment_features(features, final_bits, code, distance)


class GraphMambaDecoder(GraphDecoderBase):
    """in_channels = 2*num_cycles (same convention as the CNN)."""

    def __init__(self, in_channels=6, num_qubits=NUM_QUBITS,
                 code="heavyhex", distance=3,
                 d_model=64, num_layers=4, dropout=0.1,
                 d_state=16, d_conv=4, expand=2, fc_dim=512):
        super().__init__(in_channels, code, distance)
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is not installed - see environment.yml for the "
                "prebuilt-wheel commands")
        # node order: highest degree first (stable, fixed at construction
        # so training and inference always use the same permutation)
        deg = torch.from_numpy(self.graph.adjacency.sum(1))
        order = torch.sort(deg, descending=True, stable=True).indices
        self.register_buffer("order", order)
        self.register_buffer("inverse_order", torch.argsort(order))
        self.embedding = nn.Sequential(
            nn.Linear(IN_FEATS, d_model), nn.LayerNorm(d_model))
        self.layers = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv,
                   expand=expand) for _ in range(num_layers)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.shared = nn.Sequential(
            nn.Linear(self.num_nodes * d_model, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout))
        self.head_qubit = nn.Linear(fc_dim, num_qubits)
        self.head_logical = nn.Linear(fc_dim, 1)

    def forward(self, x):
        h = self.node_inputs(x)                          # (B, N, IN_FEATS)
        h = self.embedding(h[:, self.order])             # degree-sorted
        for mamba, norm in zip(self.layers, self.norms):
            h = norm(h + self.dropout(mamba(h)))
        h = h[:, self.inverse_order]                     # original order
        z = self.shared(h.flatten(1))
        return self.head_qubit(z), self.head_logical(z)


def compute_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                 aux_weight=0.5, qubit_pos_weight=None):
    """Return (total_loss, loss_logical, loss_qubit) - model.common."""
    return dual_head_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                          aux_weight, qubit_pos_weight)
