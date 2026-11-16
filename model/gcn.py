#!/usr/bin/env python3
"""
Dual-head GCN decoder on the detector graph.

KCS original: ../KCS/stim_simulation/models/gcn.py (class GCN).
Changes from the original:
  * two heads (per-qubit + logical) instead of one output layer: the
    original's last Linear becomes head_qubit and a parallel head_logical
    is added on the same features
  * nodes are Stim detectors from model/graph.GraphBuilder (final-Z
    detectors included) instead of (stabilizer, round) nodes; the graph
    is built inside the model, so forward takes only x
  * message passing uses the shared normalized adjacency
    self.adj = D^-1/2 (A+I) D^-1/2 (the original multiplies by the raw
    0/1 adjacency without self-loops)

Input : (B, 2*num_cycles + 1, H, W) uint8 AUGMENTED syndrome tensor
        (prepare_features). self.node_inputs(x) turns it into node
        features (B, N, IN_FEATS) = [detector bit, 6 static features].
Output: qubit_logits   (B, num_qubits) - per-qubit X-error head
                                         (diagnostics: ECR, parity_LER)
        logical_logits (B, 1)          - logical Z flip head
                                         (official metric: head-LER)

Architecture (h: node states (B, N, .), N = self.num_nodes):
  lins[0]      Linear(IN_FEATS, hidden_dim)
  lins[1..]    num_layers - 1 x Linear(hidden_dim, hidden_dim)
  each layer   h = self.adj @ ReLU(lins[i](h))
  flatten      (B, N * hidden_dim)
  head_qubit   Linear(N * hidden_dim, num_qubits)
  head_logical Linear(N * hidden_dim, 1)

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


class GCNDecoder(GraphDecoderBase):
    """in_channels = 2*num_cycles (same convention as the CNN)."""

    def __init__(self, in_channels=6, num_qubits=NUM_QUBITS,
                 code="heavyhex", distance=3,
                 hidden_dim=16, num_layers=4):
        super().__init__(in_channels, code, distance)
        # TODO: build the layers described in the module docstring
        raise NotImplementedError("GCNDecoder body is not implemented yet")

    def forward(self, x):
        h = self.node_inputs(x)                          # (B, N, IN_FEATS)
        # TODO: run the architecture -> (qubit_logits, logical_logits)
        raise NotImplementedError("GCNDecoder body is not implemented yet")


def compute_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                 aux_weight=0.5, qubit_pos_weight=None):
    """Return (total_loss, loss_logical, loss_qubit) - model.common."""
    return dual_head_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                          aux_weight, qubit_pos_weight)
