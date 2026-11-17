#!/usr/bin/env python3
"""
Triangular color code, d = 3 and d = 5 - code definition + hardware circuit
===========================================================================
Self-dual CSS code: every face carries one Z check and one X check with
the same support. Memory-Z protocol (initial |0..0>, final Z-basis data
readout); the first X-check measurement projects into the code space.
Everything distance-dependent lives in ColorDef; get one with
color_def(distance).

Labeling = the KCS hardware definition
(KCS ionq_experiment/circuits/qiskit_colorcode_generator.py):

  d = 3  [[7,1,3]] Steane code           d = 5  [[19,1,5]]
    face 0 (R) {0,1,2,3}                   face 0 (R) {1,2,5,6}
    face 1 (G) {1,2,4,5}                   face 1 (R) {3,4,7,8}
    face 2 (B) {2,3,5,6}                   face 2 (G) {0,1,5,9}
    logical Z {0,1,4}                      face 3 (G) {2,3,6,7,10,11}
    logical X {0,2,5}                      face 4 (B) {5,6,9,10,12,13}
                                           face 5 (B) {7,8,11,14}
                                           face 6 (R) {10,11,13,14,15,16}
                                           face 7 (G) {12,13,15,17}
                                           face 8 (B) {15,16,17,18}
                                           logical Z = logical X = {0,1,2,3,4}

Data coordinates (x, y) come from KCS
stim_simulation/simulation/generators/color_code.py
(COLORCODE_DATA_COORDS). The d = 5 table uses this labeling already. The
d = 3 table there is in Stim's labeling (faces {1,2,3,4}, {0,1,3,5},
{3,4,5,6}); it is relabeled here by Stim -> KCS hardware
0->0, 1->1, 2->4, 3->2, 4->5, 5->3, 6->6 (corner / edge / center roles
and the bottom edge = logical Z are preserved).

  d = 3 (data index at its coordinate)       d = 5
   y=3        6                               y=6           18
   y=2      3                                 y=5         17
   y=1        2   5                           y=4           15  16
   y=0  0   1       4                         y=3     12  13      14
        logical Z = bottom edge {0,1,4}       y=2    9      10  11
                                              y=1      5   6       7   8
                                              y=0  0  1      2  3       4

CNN tensor grid: a face sits at (row, col) = (floor(cy), floor(cx / 1.5))
of its centroid (cx, cy); the Z check and the X check of a face share the
cell (different planes).

  d = 3, GRID_SHAPE (2, 2)        d = 5, GRID_SHAPE (5, 4)
    row 1:   .   2                  row 4:  .  .  8  .
    row 0:   0   1                  row 3:  .  7  6  .
                                    row 2:  .  4  .  .
                                    row 1:  .  .  3  5
                                    row 0:  2  0  .  1

Per-cycle measurement order CYCLE_ORDER = Z0 .. Z{F-1}, X0 .. X{F-1}
(bit j of a cycle). Each check is one ancilla coupled to its face qubits
one CX at a time, in ascending data index:
    Z check: CX(data -> anc);  X check: H, CX(anc -> data), H.
This is the KCS circuit. It has no flag qubits: an ancilla fault in the
middle of a check spreads to the remaining qubits of the face. The
abstract Stim circuit (dataset_generation/color_stim.py) uses the same CX
order, so that propagation is the same in simulation.

Ancillas (IonQ is all-to-all, so there is no placement):
  "fresh"  one new ancilla per check per cycle; nothing is measured
           before the end of the circuit as far as the device is
           concerned. NUM_DATA + N_CHECKS * cycles qubits
           (d = 3, 3 cycles: 25).
  "reuse"  two ancillas (one for Z checks, one for X checks), measured
           and reset between checks. NUM_DATA + 2 qubits (d = 5: 21).
           Needs mid-circuit measurement and reset on the device.
Default: fresh for d = 3, reuse for d = 5 (fresh d = 5 with 3 cycles
would need 73 qubits). This is the KCS choice. This repository's
no-reset rule for IBM ancillas is not applied here: there each ancilla
measures a fixed check, whereas a shared no-reset ancilla would chain
the readout errors of different checks together.
In both modes a raw syndrome bit IS the check value; check_values() only
reshapes (same interface as the other codes).
"""
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# faces: (color, data indices); order = face index
_FACES = {
    3: [("R", (0, 1, 2, 3)), ("G", (1, 2, 4, 5)), ("B", (2, 3, 5, 6))],
    5: [("R", (1, 2, 5, 6)), ("R", (3, 4, 7, 8)), ("G", (0, 1, 5, 9)),
        ("G", (2, 3, 6, 7, 10, 11)), ("B", (5, 6, 9, 10, 12, 13)),
        ("B", (7, 8, 11, 14)), ("R", (10, 11, 13, 14, 15, 16)),
        ("G", (12, 13, 15, 17)), ("B", (15, 16, 17, 18))],
}
_LOGICAL_Z = {3: (0, 1, 4), 5: (0, 1, 2, 3, 4)}
_LOGICAL_X = {3: (0, 2, 5), 5: (0, 1, 2, 3, 4)}
_DATA_COORDS = {
    3: {0: (0.0, 0.0), 1: (1.0, 0.0), 4: (3.0, 0.0),
        2: (1.5, 1.0), 5: (2.5, 1.0), 3: (1.0, 2.0),
        6: (1.5, 3.0)},
    5: {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (3.0, 0.0), 3: (4.0, 0.0),
        4: (6.0, 0.0),
        5: (1.5, 1.0), 6: (2.5, 1.0), 7: (4.5, 1.0), 8: (5.5, 1.0),
        9: (1.0, 2.0), 10: (3.0, 2.0), 11: (4.0, 2.0),
        12: (1.5, 3.0), 13: (2.5, 3.0), 14: (4.5, 3.0),
        15: (3.0, 4.0), 16: (4.0, 4.0),
        17: (2.5, 5.0),
        18: (3.0, 6.0)},
}
ANCILLA_MODES = ("fresh", "reuse")
DEFAULT_ANCILLA = {3: "fresh", 5: "reuse"}


class ColorDef:
    """All distance-dependent constants of one triangular color code.

    FACES, FACE_COLORS       face index -> data indices / "R" "G" "B"
    CYCLE_ORDER              per-cycle measurement order (bit j of a cycle)
    Z_NAMES, X_NAMES, Z_POS  names per type, positions of Z in CYCLE_ORDER
    CHECK_DEFS               name -> (type, support data indices, face
                             index, color)
    DIDX                     data index -> data index (identity; the
                             data "coordinate" of this code is its index)
    LOGICAL_Z_IDX, LOGICAL_X_IDX
    DATA_COORDS              data index -> (x, y)
    GRID_SHAPE, ANC_GRID     CNN tensor grid; name -> (row, col)"""

    def __init__(self, distance):
        d = int(distance)
        if d not in _FACES:
            raise ValueError(f"color distance must be one of "
                             f"{sorted(_FACES)}, got {d}")
        self.DISTANCE = d
        self.FACES = [tuple(q) for _, q in _FACES[d]]
        self.FACE_COLORS = [c for c, _ in _FACES[d]]
        self.N_FACES = len(self.FACES)
        self.N_CHECKS = 2 * self.N_FACES
        self.NUM_DATA = 1 + max(q for f in self.FACES for q in f)
        self.DATA_COORDS = dict(_DATA_COORDS[d])
        assert sorted(self.DATA_COORDS) == list(range(self.NUM_DATA))
        self.DIDX = {q: q for q in range(self.NUM_DATA)}

        self.Z_NAMES = [f"Z{i}" for i in range(self.N_FACES)]
        self.X_NAMES = [f"X{i}" for i in range(self.N_FACES)]
        self.CYCLE_ORDER = self.Z_NAMES + self.X_NAMES
        self.Z_POS = list(range(self.N_FACES))
        self.CHECK_DEFS = {}
        for i, face in enumerate(self.FACES):
            for t in ("Z", "X"):
                self.CHECK_DEFS[f"{t}{i}"] = (t, face, i,
                                              self.FACE_COLORS[i])
        self.Z_STABS = {n: self.CHECK_DEFS[n][1] for n in self.Z_NAMES}
        self.X_STABS = {n: self.CHECK_DEFS[n][1] for n in self.X_NAMES}
        self.LOGICAL_Z_IDX = list(_LOGICAL_Z[d])
        self.LOGICAL_X_IDX = list(_LOGICAL_X[d])

        cells = []
        for face in self.FACES:
            cx = sum(self.DATA_COORDS[q][0] for q in face) / len(face)
            cy = sum(self.DATA_COORDS[q][1] for q in face) / len(face)
            cells.append((math.floor(cy), math.floor(cx / 1.5)))
        assert len(set(cells)) == self.N_FACES, "face cells collide"
        self.GRID_SHAPE = (1 + max(r for r, _ in cells),
                           1 + max(c for _, c in cells))
        self.ANC_GRID = {f"{t}{i}": cells[i]
                         for i in range(self.N_FACES) for t in ("Z", "X")}

        # sanity: self-dual CSS code with one logical qubit
        h = self.parity_check()
        assert not ((h @ h.T) % 2).any(), "faces do not commute"
        lz = np.zeros(self.NUM_DATA, dtype=int)
        lz[self.LOGICAL_Z_IDX] = 1
        lx = np.zeros(self.NUM_DATA, dtype=int)
        lx[self.LOGICAL_X_IDX] = 1
        assert not ((h @ lz) % 2).any() and not ((h @ lx) % 2).any()
        assert (lz @ lx) % 2 == 1
        # every qubit is in at most one face of each color
        for color in "RGB":
            rows = [i for i, c in enumerate(self.FACE_COLORS) if c == color]
            assert h[rows].sum(axis=0).max() <= 1

    def parity_check(self):
        """(N_FACES, NUM_DATA) int matrix of the face supports (H_z = H_x)."""
        h = np.zeros((self.N_FACES, self.NUM_DATA), dtype=int)
        for i, face in enumerate(self.FACES):
            h[i, list(face)] = 1
        return h

    def num_qubits(self, num_cycles, ancilla=None):
        ancilla = ancilla or DEFAULT_ANCILLA[self.DISTANCE]
        if ancilla == "fresh":
            return self.NUM_DATA + self.N_CHECKS * num_cycles
        return self.NUM_DATA + 2


_DEFS = {}


def color_def(distance=3):
    """ColorDef of a distance (built once)."""
    distance = int(distance)
    if distance not in _DEFS:
        _DEFS[distance] = ColorDef(distance)
    return _DEFS[distance]


for _d in (3, 5):                       # build + self-check at import
    color_def(_d)


def check_values(syn_rows, num_cycles, distance=3):
    """Raw syndrome bits -> per-cycle check values {name: (shots, C)}.
    Every ancilla starts in |0> (fresh or reset), so a raw bit is the
    check value."""
    S = color_def(distance)
    syn_rows = np.asarray(syn_rows)
    return {name: np.stack([syn_rows[:, cyc * S.N_CHECKS + j]
                            for cyc in range(num_cycles)],
                           axis=1).astype(int)
            for j, name in enumerate(S.CYCLE_ORDER)}


# ==================================================================
# hardware circuit (Qiskit) - memory-Z
# ==================================================================
class ColorCodeHardware:
    """Color code circuit in local qubit indices: data 0 .. NUM_DATA-1,
    then the ancillas (module docstring).

    ancilla      "fresh" | "reuse" (default: DEFAULT_ANCILLA[distance])
    inject:      (pauli, data index, after_cycle) - data error between
                 cycles (verification)

    Classical registers match the other codes: syn[cyc * N_CHECKS + j]
    (j = position in CYCLE_ORDER), then data[NUM_DATA]."""

    def __init__(self, num_cycles=3, distance=3, ancilla=None):
        self.num_cycles = num_cycles
        self.distance = distance
        self.ancilla = ancilla or DEFAULT_ANCILLA[int(distance)]
        if self.ancilla not in ANCILLA_MODES:
            raise ValueError(f"ancilla must be one of {ANCILLA_MODES}")

    def build_circuit(self, initial_state=0, inject=None):
        S = color_def(self.distance)
        n = S.NUM_DATA
        q = QuantumRegister(S.num_qubits(self.num_cycles, self.ancilla), 'q')
        syn = ClassicalRegister(S.N_CHECKS * self.num_cycles, 'syn')
        dat = ClassicalRegister(n, 'data')
        qc = QuantumCircuit(q, syn, dat)
        if initial_state == 1:
            for i in S.LOGICAL_X_IDX:
                qc.x(i)
        used = set()                    # reuse mode: ancillas used so far
        for cyc in range(self.num_cycles):
            for j, name in enumerate(S.CYCLE_ORDER):
                ctype, support = S.CHECK_DEFS[name][0], S.CHECK_DEFS[name][1]
                if self.ancilla == "fresh":
                    a = n + cyc * S.N_CHECKS + j
                else:
                    a = n + (0 if ctype == "Z" else 1)
                    if a in used:
                        qc.reset(a)
                    used.add(a)
                if ctype == "Z":
                    for i in support:
                        qc.cx(i, a)
                else:
                    qc.h(a)
                    for i in support:
                        qc.cx(a, i)
                    qc.h(a)
                qc.measure(a, syn[cyc * S.N_CHECKS + j])
            qc.barrier()
            if inject is not None and inject[2] == cyc:
                pauli, dq, _ = inject
                getattr(qc, pauli.lower())(dq)
                qc.barrier()
        for i in range(n):
            qc.measure(i, dat[i])
        return qc


if __name__ == '__main__':
    for d in (3, 5):
        S = color_def(d)
        print(f"d={d}: data={S.NUM_DATA} faces={S.N_FACES} "
              f"grid={S.GRID_SHAPE} cells={ {i: S.ANC_GRID[f'Z{i}'] for i in range(S.N_FACES)} }")
        for mode in ANCILLA_MODES:
            qc = ColorCodeHardware(3, d, mode).build_circuit()
            ops = qc.count_ops()
            print(f"  {mode}: qubits={qc.num_qubits} cx={ops.get('cx')} "
                  f"h={ops.get('h')} meas={ops.get('measure')} "
                  f"reset={ops.get('reset', 0)}")
