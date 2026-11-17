#!/usr/bin/env python3
"""
Color code MWPM baseline: restriction decoder (d = 3, 5)
========================================================
The DEM route of baseline/mwpm.py does not carry over to the color code:
a data error flips up to three faces, and Stim cannot decompose the d = 5
detector error model into graphlike errors (decompose_errors fails).
This is the KCS decoder instead
(KCS ionq_experiment/decoders/mwpm_colorcode_decoder.py,
MWPMColorCodeDecoder), a per-qubit decoder of ONE Z syndrome:

  1. for each color pair (RG, RB, GB) keep only the faces of those two
     colors; every data qubit is then in at most two faces, so
     pymatching.Matching(sub H) applies; decode the restricted syndrome
  2. keep the corrections that reproduce the FULL syndrome and take the
     lightest one
  3. if none does, fall back to the single-error lookup (a correction of
     one qubit whose face pattern equals the syndrome, else no
     correction)

The decoder depends only on the syndrome, and there are 2^F syndromes
(8 for d = 3, 512 for d = 5), so the constructor tabulates the answer for
every syndrome once and decode() is a table lookup.

As a baseline the decoder reads the Z syndrome of the final data readout
(H_z . final bits) and predicts the logical flip as the parity of its
correction over logical Z. It does not use the syndrome history.
"""
import sys
from pathlib import Path

import numpy as np
import pymatching

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from circuits.color.colorcode import color_def  # noqa: E402
from evaluation.metrics import ler  # noqa: E402

COLOR_PAIRS = (("R", "G"), ("R", "B"), ("G", "B"))


class RestrictionDecoder:
    """Per-qubit color-code decoder of a Z (face) syndrome.

    H          (num_faces, num_qubits) uint8
    logical_z  data indices of logical Z
    table      (2^F, num_qubits) uint8: correction of syndrome index
               sum_i s_i << i
    consistent (2^F,) bool: the tabulated correction reproduces the
               syndrome"""

    def __init__(self, distance=3):
        S = color_def(distance)
        self.distance = S.DISTANCE
        self.num_qubits = S.NUM_DATA
        self.H = S.parity_check().astype(np.uint8)
        self.logical_z = list(S.LOGICAL_Z_IDX)
        f = S.N_FACES
        self._weights = (1 << np.arange(f)).astype(np.int64)

        self._restrictions = []
        for c1, c2 in COLOR_PAIRS:
            rows = [i for i, c in enumerate(S.FACE_COLORS) if c in (c1, c2)]
            self._restrictions.append(
                (rows, pymatching.Matching(self.H[rows])))
        self._single = {tuple(self.H[:, q].tolist()): q
                        for q in range(self.num_qubits)}

        self.table = np.zeros((1 << f, self.num_qubits), dtype=np.uint8)
        for k in range(1 << f):
            s = ((k >> np.arange(f)) & 1).astype(np.uint8)
            self.table[k] = self._decode_one(s)
        syn = (self.table.astype(np.int64) @ self.H.T.astype(np.int64)) & 1
        self.consistent = (syn @ self._weights) == np.arange(1 << f)

    def _decode_one(self, s):
        corr = np.zeros(self.num_qubits, dtype=np.uint8)
        if not s.any():
            return corr
        best = None
        for rows, matching in self._restrictions:
            if not s[rows].any():
                continue
            c = np.asarray(matching.decode(s[rows]), dtype=np.uint8)
            if np.array_equal((self.H @ c) % 2, s) and (
                    best is None or c.sum() < best.sum()):
                best = c
        if best is not None:
            return best
        q = self._single.get(tuple(s.tolist()))
        if q is not None:
            corr[q] = 1
        return corr

    def syndrome(self, bits):
        """(shots, num_qubits) data bits -> (shots, num_faces)."""
        bits = np.asarray(bits).astype(np.int64)
        return ((bits @ self.H.T.astype(np.int64)) & 1).astype(np.uint8)

    def decode_batch(self, syndrome):
        """(shots, num_faces) -> (shots, num_qubits) correction mask."""
        syndrome = np.asarray(syndrome).astype(np.int64)
        return self.table[syndrome @ self._weights]

    def logical(self, correction):
        c = np.asarray(correction).astype(np.int64)
        return (c[:, self.logical_z].sum(axis=1) & 1).astype(np.uint8)

    def predict(self, final_bits):
        """Final data bits -> predicted logical Z flips (baseline)."""
        return self.logical(self.decode_batch(self.syndrome(final_bits)))


def decode_ler(decoder, final_bits, true_logical):
    """(LER, predicted flips) of the restriction decoder on the final
    data syndrome."""
    pred = decoder.predict(final_bits)
    return ler(pred, true_logical), pred
