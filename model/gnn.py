#!/usr/bin/env python3
"""
Dual-head GNN decoder.

Graph : nodes = Stim detectors (order of the code's _append_detectors;
        heavyhex (3,3) cycles=3 has 48 nodes incl. the 8 final-Z
        detectors — same input as MWPM), static edges from
        model/graph.GraphBuilder (spatial support-sharing / temporal /
        final-Z links).
Input : (B, 2*num_cycles + 1, H, W) uint8 AUGMENTED syndrome tensor —
        the dataset tensor plus one channel with the final-Z detector
        bits (model/graph.augment_features; train.py / hardware/run_hw.py
        apply it through prepare_features).
Output: qubit_logits   (B, num_qubits) — per-qubit X-error head
                                         (diagnostics: ECR, parity_LER)
        logical_logits (B, 1)          — logical Z flip head
                                         (official metric: head-LER)

Interface used by train.py and hardware/run_hw.py:
  * GNNDecoder(in_channels, num_qubits, code, distance, ...) with
    forward(x) -> (qubit_logits, logical_logits)
    (in_channels = 2*num_cycles, same convention as the CNN; the model
     derives num_cycles = in_channels // 2)
  * compute_loss(...) -> (total_loss, loss_logical, loss_qubit)
    The loss is LER-first: BCE on the logical head is the MAIN loss and
    BCE on the per-qubit head is an AUXILIARY loss scaled by aux_weight
    (default 0.5).
  * prepare_features(features, final_bits, code, distance)

Pure torch only — no torch_geometric or other graph libraries. The graph
is small, so message passing is a dense matmul with the normalized
adjacency buffer `self.adj`: neighbor aggregation of node states
h (B, N, H) is `self.adj @ h`.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.graph import GraphBuilder, augment_features, IN_FEATS  # noqa: E402

NUM_QUBITS = 17


def prepare_features(features, final_bits, code="heavyhex", distance=3):
    """Dataset tensor (B,2C,H,W) + final bits (B,num_qubits) -> augmented input.

    train.py passes the per-qubit flip labels; hardware/run_hw.py passes
    the measured final data bits. Same detector bits either way (see
    model/graph.py)."""
    return augment_features(features, final_bits, code, distance)


class GNNDecoder(nn.Module):
    """in_channels = 2*num_cycles (same convention as the CNN)."""

    def __init__(self, in_channels=6, num_qubits=NUM_QUBITS,
                 code="heavyhex", distance=3,
                 hidden_dim=128, num_layers=4, fc_dim=256, dropout=0.1):
        super().__init__()
        # static graph buffers
        num_cycles = in_channels // 2
        gb = GraphBuilder(code, num_cycles, distance)
        self.graph = gb
        self.register_buffer("adj",
                             torch.from_numpy(gb.norm_adjacency))
        self.register_buffer("static_feats",
                             torch.from_numpy(gb.static_feats))
        raise NotImplementedError("GNNDecoder body is not implemented yet")

    def forward(self, x):
        # detector bits + static features per node
        det = self.graph.node_values(x).float()          # (B, N)
        stat = self.static_feats.unsqueeze(0).expand(x.shape[0], -1, -1)
        h = torch.cat([det.unsqueeze(-1), stat], dim=-1)  # (B, N, IN_FEATS)
        raise NotImplementedError("GNNDecoder body is not implemented yet")


def compute_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                 aux_weight=0.5, qubit_pos_weight=None):
    """Return (total_loss, loss_logical, loss_qubit).

    total = BCE(logical_logits, y_logical) + aux_weight * BCE(qubit_logits,
    y_qubit). qubit_pos_weight (optional, shape (num_qubits,)) goes into
    the per-qubit BCE's pos_weight to counter class imbalance
    (e.g. pos_weight=(1-p)/p).
    """
    raise NotImplementedError("compute_loss is not implemented yet")
