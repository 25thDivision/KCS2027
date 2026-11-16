#!/usr/bin/env python3
"""
Dual-head Graph Transformer decoder on the detector graph.

KCS original: ../KCS/stim_simulation/models/graph_transformer.py (class GraphTransformer).
Changes from the original:
  * two heads (per-qubit + logical) instead of one output layer: the
    original's last Linear becomes head_qubit and a parallel head_logical
    is added on the same features
  * nodes are Stim detectors from model/graph.GraphBuilder (final-Z
    detectors included) instead of (stabilizer, round) nodes; the graph
    is built inside the model, so forward takes only x
  * the attention mask is a buffer built once from self.adj_mask; the
    original rebuilds it from edge_index on every forward

Input : (B, 2*num_cycles + 1, H, W) uint8 AUGMENTED syndrome tensor
        (prepare_features). self.node_inputs(x) turns it into node
        features (B, N, IN_FEATS) = [detector bit, 6 static features].
Output: qubit_logits   (B, num_qubits) - per-qubit X-error head
                                         (diagnostics: ECR, parity_LER)
        logical_logits (B, 1)          - logical Z flip head
                                         (official metric: head-LER)

Architecture (h: node states (B, N, .), N = self.num_nodes):
  embedding    Linear(IN_FEATS, d_model)
  layers       num_layers x post-norm transformer layer, as the original
               GraphTransformerLayer:
                 a = MultiheadAttention(d_model, num_heads, dropout=dropout,
                     batch_first=True)(h, h, h, attn_mask=self.attn_mask,
                     need_weights=False)
                 h = LayerNorm(h + Dropout(a))
                 f = Linear(4*d_model, d_model)(Dropout(ReLU(
                     Linear(d_model, 4*d_model)(h))))
                 h = LayerNorm(h + Dropout(f))
               self.attn_mask (N, N) float: 0 where attention is allowed,
               -inf elsewhere (provided buffer)
  flatten      (B, N * d_model)
  shared       Linear(N * d_model, 2 * d_model) -> ReLU -> Dropout(dropout)
  head_qubit   Linear(2 * d_model, num_qubits)
  head_logical Linear(2 * d_model, 1)

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


class GraphTransformerDecoder(GraphDecoderBase):
    """in_channels = 2*num_cycles (same convention as the CNN)."""

    def __init__(self, in_channels=6, num_qubits=NUM_QUBITS,
                 code="heavyhex", distance=3,
                 d_model=256, num_heads=4, num_layers=5, dropout=0.1):
        super().__init__(in_channels, code, distance)
        # attention mask: 0 where allowed (neighbors + self), -inf elsewhere
        mask = torch.zeros(self.num_nodes, self.num_nodes)
        mask.masked_fill_(~self.adj_mask, float("-inf"))
        self.register_buffer("attn_mask", mask)
        # TODO: build the layers described in the module docstring
        raise NotImplementedError("GraphTransformerDecoder body is not implemented yet")

    def forward(self, x):
        h = self.node_inputs(x)                          # (B, N, IN_FEATS)
        # TODO: run the architecture -> (qubit_logits, logical_logits)
        raise NotImplementedError("GraphTransformerDecoder body is not implemented yet")


def compute_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                 aux_weight=0.5, qubit_pos_weight=None):
    """Return (total_loss, loss_logical, loss_qubit) - model.common."""
    return dual_head_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                          aux_weight, qubit_pos_weight)
