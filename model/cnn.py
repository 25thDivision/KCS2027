#!/usr/bin/env python3
"""
Dual-head CNN decoder.

KCS original: ../KCS/stim_simulation/models/cnn.py (class CNN).
Changes from the original:
  * two heads (per-qubit + logical) instead of one output layer
  * input is the (2*cycles + 1)-channel augmented syndrome tensor instead
    of a 1-channel image
  * a stack of conv blocks with BatchNorm (the original has one 32-channel
    conv), wider shared FC, dropout

Input : (B, 2*num_cycles + 1, H, W) uint8 tensor
        - heavyhex: 4x5 diamond embedding of the 8 ancillas
          (dataset_generation/heavyhex33_stim.py, ANC_COORD)
        - surface: 4x4 plaquette-vertex grid
          (circuits/rotatedSurface/rotatedSurface3.py, ANC_GRID)
        - the (code, distance) pair selects the grid via model.code_spec
        - channels alternate [Z-plane, X-plane] per cycle; the LAST
          channel holds the final-Z detector bits (prepare_features /
          model.graph.augment_features - the same detector set the graph
          models and MWPM consume). Not label leakage: logical Z is not a
          product of Z-stabilizers.
Output: qubit_logits   (B, num_qubits) - per-qubit X-error head
                                         (diagnostics: ECR, parity_LER)
        logical_logits (B, 1)          - logical Z flip head
                                         (official metric: head-LER)

Architecture:
  features     num_blocks x [Conv2d(3x3, padding=1) -> BatchNorm2d -> ReLU];
               first conv takes in_channels + 1 channels, every conv
               outputs conv_channels; no pooling (grid size is kept)
  shared       Flatten -> Linear(conv_channels*H*W, fc_dim) -> ReLU
               -> Dropout(dropout)
  head_qubit   Linear(fc_dim, num_qubits)
  head_logical Linear(fc_dim, 1)
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.common import dual_head_loss  # noqa: E402
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
                 conv_channels=64, num_blocks=3, fc_dim=256, dropout=0.1):
        super().__init__()
        from model import code_spec
        grid_h, grid_w = code_spec(code, distance)["grid"]
        # TODO: build self.features, self.shared, self.head_qubit,
        # self.head_logical as described in the module docstring
        raise NotImplementedError("CNNDecoder body is not implemented yet")

    def forward(self, x):
        x = x.float()
        # TODO: features -> shared -> (qubit_logits, logical_logits)
        raise NotImplementedError("CNNDecoder body is not implemented yet")


def compute_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                 aux_weight=0.5, qubit_pos_weight=None):
    """Return (total_loss, loss_logical, loss_qubit) - model.common."""
    return dual_head_loss(qubit_logits, logical_logits, y_qubit, y_logical,
                          aux_weight, qubit_pos_weight)
