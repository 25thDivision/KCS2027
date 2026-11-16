#!/usr/bin/env python3
"""
Dual-head CNN decoder.

Input : (B, 2*num_cycles + 1, H, W) uint8 tensor
        - heavyhex: 4x5 diamond embedding of the 8 ancillas
          (dataset_generation/heavyhex33_stim.py, ANC_COORD)
        - surface: 4x4 plaquette-vertex grid
          (circuits/rotatedSurface/rotatedSurface3.py, ANC_GRID)
        - the (code, distance) pair selects the grid via model.code_spec
        - channels alternate [Z-plane, X-plane] per cycle
          (default num_cycles=3 -> in_channels=6, so the tensor the
          model actually sees has 7 channels)
        - the LAST channel holds the final-Z detector bits
          (prepare_features / model.graph.augment_features — the same
          detector set the graph models and MWPM consume). Not label
          leakage: logical Z is not a product of Z-stabilizers, so
          head-LER's label cannot be derived from it.
Output: qubit_logits   (B, num_qubits) — per-qubit X-error head
                                         (diagnostics: ECR, parity_LER)
        logical_logits (B, 1)          — logical Z flip head
                                         (official metric: head-LER)

Interface used by train.py and hardware/run_hw.py:
  * CNNDecoder(in_channels, num_qubits, code, distance, ...) with
    forward(x) -> (qubit_logits, logical_logits)
  * compute_loss(...) -> (total_loss, loss_logical, loss_qubit)
    The loss is LER-first: BCE on the logical head is the MAIN loss and
    BCE on the per-qubit head is an AUXILIARY loss scaled by aux_weight
    (default 0.5).
  * prepare_features(features, final_bits, code, distance)
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.graph import augment_features  # noqa: E402

NUM_QUBITS = 17


def prepare_features(features, final_bits, code="heavyhex", distance=3):
    """Dataset tensor (B,2C,H,W) + final bits (B,num_qubits) -> augmented input.

    train.py passes the per-qubit flip labels; hardware/run_hw.py passes
    the measured final data bits. Same detector bits either way (see
    model/graph.py)."""
    return augment_features(features, final_bits, code, distance)


class CNNDecoder(nn.Module):
    def __init__(self, in_channels=6, num_qubits=NUM_QUBITS,
                 code="heavyhex", distance=3,
                 conv_channels=64, fc_dim=256, dropout=0.1):
        super().__init__()
        from model import code_spec
        grid_h, grid_w = code_spec(code, distance)["grid"]
        raise NotImplementedError("CNNDecoder body is not implemented yet")

    def forward(self, x):
        raise NotImplementedError("CNNDecoder body is not implemented yet")


def compute_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                 aux_weight=0.5, qubit_pos_weight=None):
    """Return (total_loss, loss_logical, loss_qubit).

    total = BCE(logical_logits, y_logical) + aux_weight * BCE(qubit_logits,
    y_qubit). qubit_pos_weight (optional, shape (num_qubits,)) goes into
    the per-qubit BCE's pos_weight to counter class imbalance
    (e.g. pos_weight=(1-p)/p).
    """
    raise NotImplementedError("compute_loss is not implemented yet")
