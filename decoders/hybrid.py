#!/usr/bin/env python3
"""
Hybrid decoders: MWPM + ML and ML + MWPM
========================================
baseline/mwpm.py predicts only the observable flip, so the hybrids use a
per-qubit MWPM on the Z syndrome of the FINAL data readout:

    s = H_z . final_bits          (simulation: npz "labels", the final
                                   measurement flips; hardware: the
                                   measured final data bits)
    Matching(H_z).decode(s) -> X-correction mask over the data qubits

H_z comes from the code definition (model.graph.code_adapter: the
Z-check supports in final-Z detector order).

  MWPM+ML   c1 = Matching(H_z).decode(s)
            r  = s ^ H_z . c1
            c2 = (qubit head > 0) on an input that is zero except for r
                 in the final-Z detector slots; shots with r = 0 skip
                 the model (c2 = 0)
  ML+MWPM   c1 = (qubit head > 0) on the real input
            r  = s ^ H_z . c1
            c2 = Matching(H_z).decode(r)
  both      logical flip = parity of (c1 ^ c2) over LOGICAL_Z

Matching(H_z) returns a correction with H_z . c = s whenever s is
matchable, and every syndrome of the codes here is (each has weight-1
columns, i.e. a boundary). So in MWPM+ML the residual r is always 0, the
model is never consulted, and MWPM+ML equals the final-syndrome MWPM
alone for every model. In ML+MWPM the matching completes the model's
correction to one that reproduces the final syndrome.
"""
import sys
from pathlib import Path

import numpy as np
import pymatching

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.metrics import logical_qubit_indices  # noqa: E402
from model import code_spec  # noqa: E402
from model.graph import code_adapter  # noqa: E402


class FinalSyndromeMatcher:
    """Per-qubit MWPM on the final-data Z syndrome of one (code, distance).

    H          (num_z_checks, num_qubits) uint8, rows in final-Z
               detector order
    logical_z  data indices of logical Z"""

    def __init__(self, code, distance=3):
        self.code, self.distance = code, distance
        ad = code_adapter(code, distance)
        self.num_qubits = code_spec(code, distance)["num_qubits"]
        self.z_names = [ad["check_at"][j] for j in ad["z_pos"]]
        self.cells = [ad["coord"][n] for n in self.z_names]
        self.grid = ad["grid"]
        self.H = np.zeros((len(self.z_names), self.num_qubits),
                          dtype=np.uint8)
        for i, name in enumerate(self.z_names):
            self.H[i, ad["support_idx"][name]] = 1
        self.logical_z = list(logical_qubit_indices(code, distance))
        self.matching = pymatching.Matching(self.H)

    def syndrome(self, bits):
        """(shots, num_qubits) data bits -> (shots, num_z_checks)."""
        bits = np.asarray(bits).astype(np.int64)
        return ((bits @ self.H.T.astype(np.int64)) & 1).astype(np.uint8)

    def decode(self, syndrome):
        """(shots, num_z_checks) -> (shots, num_qubits) correction mask."""
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        if syndrome.shape[0] == 0:
            return np.zeros((0, self.num_qubits), dtype=np.uint8)
        return np.asarray(self.matching.decode_batch(syndrome),
                          dtype=np.uint8)

    def logical(self, correction):
        """(shots, num_qubits) mask -> (shots,) logical Z flip."""
        c = np.asarray(correction).astype(np.int64)
        return (c[:, self.logical_z].sum(axis=1) & 1).astype(np.uint8)

    def residual_input(self, residual, num_cycles):
        """Residual syndrome -> model input (shots, 2C+1, H, W) uint8:
        all zero except the residual in the final-Z detector slots (the
        last channel, at the Z-ancilla cells)."""
        residual = np.asarray(residual, dtype=np.uint8)
        x = np.zeros((residual.shape[0], 2 * num_cycles + 1, *self.grid),
                     dtype=np.uint8)
        for i, (r, c) in enumerate(self.cells):
            x[:, -1, r, c] = residual[:, i]
        return x


def qubit_correction(model, x, batch_size=8192):
    """Model-ready input (numpy or torch, uint8) -> (shots, num_qubits)
    uint8 mask from the per-qubit head (logit > 0)."""
    import torch
    device = next(model.parameters()).device
    x = torch.as_tensor(x)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            ql, _ = model(x[i:i + batch_size].to(device))
            out.append((ql.float().cpu().numpy() > 0).astype(np.uint8))
    return np.concatenate(out)


def hybrid_predictions(model, mod, features, final_bits, code, distance=3,
                       matcher=None, q_logits=None, batch_size=8192):
    """Logical-flip predictions of both hybrids.

    features    (shots, 2C, H, W) uint8 syndrome tensor (not augmented)
    final_bits  (shots, num_qubits) final data bits / flip labels
    q_logits    optional (shots, num_qubits) per-qubit logits already
                computed on the real input (skips one model pass)
    Returns {"mwpm_ml": (shots,), "ml_mwpm": (shots,), "mwpm": (shots,)};
    "mwpm" is the final-syndrome MWPM alone (stage 1 of MWPM+ML)."""
    matcher = matcher or FinalSyndromeMatcher(code, distance)
    features = np.asarray(features, dtype=np.uint8)
    final_bits = np.asarray(final_bits, dtype=np.uint8)
    num_cycles = features.shape[1] // 2
    s = matcher.syndrome(final_bits)

    # MWPM + ML
    c1 = matcher.decode(s)
    r = s ^ matcher.syndrome(c1)
    todo = np.nonzero(r.any(axis=1))[0]
    combined = c1.copy()
    if todo.size:
        combined[todo] ^= qubit_correction(
            model, matcher.residual_input(r[todo], num_cycles), batch_size)
    mwpm_ml = matcher.logical(combined)

    # ML + MWPM
    if q_logits is None:
        x = mod.prepare_features(features, final_bits, code, distance)
        m1 = qubit_correction(model, x, batch_size)
    else:
        m1 = (np.asarray(q_logits) > 0).astype(np.uint8)
    ml_mwpm = matcher.logical(m1 ^ matcher.decode(s ^ matcher.syndrome(m1)))

    return {"mwpm_ml": mwpm_ml, "ml_mwpm": ml_mwpm,
            "mwpm": matcher.logical(c1)}
