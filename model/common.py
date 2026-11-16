#!/usr/bin/env python3
"""
Building blocks shared by the decoder models
============================================
* dual_head_loss   - the LER-first two-head loss every model module
                     exposes as compute_loss
* GraphDecoderBase - static detector-graph buffers and the per-node input
                     features of the graph decoders (model/graph.py)
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.graph import GraphBuilder, IN_FEATS  # noqa: E402,F401


def dual_head_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                   aux_weight=0.5, qubit_pos_weight=None):
    """Return (total_loss, loss_logical, loss_qubit).

    total = BCE(logical_logits, y_logical) + aux_weight * BCE(qubit_logits,
    y_qubit). The logical head is the MAIN loss, the per-qubit head the
    AUXILIARY one. qubit_pos_weight (optional, shape (num_qubits,)) goes
    into the per-qubit BCE's pos_weight to counter class imbalance
    (e.g. pos_weight=(1-p)/p). Labels may be uint8."""
    loss_logical = F.binary_cross_entropy_with_logits(
        logical_logits.reshape(-1).float(), y_logical.reshape(-1).float())
    loss_qubit = F.binary_cross_entropy_with_logits(
        qubit_logits.float(), y_qubit.float(), pos_weight=qubit_pos_weight)
    return loss_logical + aux_weight * loss_qubit, loss_logical, loss_qubit


class GraphDecoderBase(nn.Module):
    """Graph plumbing of a detector-graph decoder.

    Buffers (N = number of detector nodes):
      adj           (N, N) float - D^-1/2 (A+I) D^-1/2; neighbor
                    aggregation of node states h (B, N, H) is `self.adj @ h`
      adj_mask      (N, N) bool  - True where two nodes are adjacent or
                    identical (attention may look there)
      static_feats  (N, 6) float - static per-node features
    self.graph is the GraphBuilder (raw 0/1 `adjacency`, node list)."""

    def __init__(self, in_channels, code, distance):
        super().__init__()
        gb = GraphBuilder(code, in_channels // 2, distance)
        self.graph = gb
        self.num_nodes = gb.num_nodes
        self.register_buffer("adj", torch.from_numpy(gb.norm_adjacency))
        self.register_buffer(
            "adj_mask",
            torch.from_numpy(gb.adjacency > 0)
            | torch.eye(gb.num_nodes, dtype=torch.bool))
        self.register_buffer("static_feats",
                             torch.from_numpy(gb.static_feats))

    def node_inputs(self, x):
        """Augmented tensor (B, 2C+1, H, W) -> node features (B, N, IN_FEATS):
        [detector bit, 6 static features] per node."""
        det = self.graph.node_values(x).float()          # (B, N)
        stat = self.static_feats.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([det.unsqueeze(-1), stat], dim=-1)
