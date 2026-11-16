#!/usr/bin/env python3
"""
Dual-head GAT decoder on the detector graph.

KCS original: ../KCS/stim_simulation/models/gat.py (class GAT).
Changes from the original:
  * two heads (per-qubit + logical) instead of one output layer: the
    original's last Linear becomes head_qubit and a parallel head_logical
    is added on the same features
  * nodes are Stim detectors from model/graph.GraphBuilder (final-Z
    detectors included) instead of (stabilizer, round) nodes; the graph
    is built inside the model, so forward takes only x
  * attention is masked with self.adj_mask (neighbors and the node
    itself); the original masks with the raw adjacency, so a node could
    not attend to itself
  * num_layers attention layers are stacked (the original has one)

Input : (B, 2*num_cycles + 1, H, W) uint8 AUGMENTED syndrome tensor
        (prepare_features). self.node_inputs(x) turns it into node
        features (B, N, IN_FEATS) = [detector bit, 6 static features].
Output: qubit_logits   (B, num_qubits) - per-qubit X-error head
                                         (diagnostics: ECR, parity_LER)
        logical_logits (B, 1)          - logical Z flip head
                                         (official metric: head-LER)

Architecture (h: node states (B, N, .), N = self.num_nodes):
  layers       num_layers x GATLayer, each with its own parameters; the
               first takes IN_FEATS features, the rest hidden_dim. ELU
               between layers, none after the last one.
               One GATLayer is the whole attention body of the original:
    lin_in       Linear(in, hidden_dim); split into `heads` heads of
                 hidden_dim // heads features: (B, N, heads, hd)
    att_src, att_dst   Parameter (1, heads, hd) each, xavier_uniform
    scores       LeakyReLU(0.2)(a_src[i] + a_dst[j]) per head, shape
                 (B, N, N, heads) with a_* = (h * att_*).sum(-1);
                 positions where self.adj_mask is False are filled with
                 -1e9; softmax over j; dropout(p=dropout) on the weights
    aggregate    out[i] = sum_j alpha[i, j] * h[j] per head, heads
                 concatenated back to (B, N, hidden_dim)
  flatten      (B, N * hidden_dim)
  head_qubit   Linear(N * hidden_dim, num_qubits)
  head_logical Linear(N * hidden_dim, 1)
  The original accepts num_layers but applies a single attention layer;
  this port stacks num_layers of them (default 4).

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


class GATLayer(nn.Module):
    """One multi-head graph attention layer (the KCS GAT body)."""

    def __init__(self, in_dim, hidden_dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.dropout = dropout
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.att_src = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.head_dim))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, h, adj_mask):
        B, N, _ = h.shape
        h = self.lin_in(h).view(B, N, self.heads, self.head_dim)
        a_src = (h * self.att_src).sum(-1)               # (B, N, heads)
        a_dst = (h * self.att_dst).sum(-1)
        scores = F.leaky_relu(a_src.unsqueeze(2) + a_dst.unsqueeze(1), 0.2)
        scores = scores.masked_fill(~adj_mask[None, :, :, None], -1e9)
        alpha = F.softmax(scores, dim=2)                 # over neighbors j
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.einsum("bijh,bjhd->bihd", alpha, h)
        return out.reshape(B, N, -1)                     # (B, N, hidden)


class GATDecoder(GraphDecoderBase):
    """in_channels = 2*num_cycles (same convention as the CNN)."""

    def __init__(self, in_channels=6, num_qubits=NUM_QUBITS,
                 code="heavyhex", distance=3,
                 hidden_dim=16, heads=4, num_layers=4, dropout=0.1):
        super().__init__(in_channels, code, distance)
        if hidden_dim % heads:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible "
                             f"by heads ({heads})")
        self.layers = nn.ModuleList(
            [GATLayer(IN_FEATS if i == 0 else hidden_dim, hidden_dim, heads,
                      dropout) for i in range(num_layers)])
        self.head_qubit = nn.Linear(self.num_nodes * hidden_dim, num_qubits)
        self.head_logical = nn.Linear(self.num_nodes * hidden_dim, 1)

    def forward(self, x):
        h = self.node_inputs(x)                          # (B, N, IN_FEATS)
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h = layer(h, self.adj_mask)
            if i < last:
                h = F.elu(h)
        z = h.flatten(1)                                 # (B, N * hidden)
        return self.head_qubit(z), self.head_logical(z)


def compute_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                 aux_weight=0.5, qubit_pos_weight=None):
    """Return (total_loss, loss_logical, loss_qubit) - model.common."""
    return dual_head_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                          aux_weight, qubit_pos_weight)
