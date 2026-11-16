#!/usr/bin/env python3
"""
Rotated surface code, odd distance d - code definition + hardware circuit
=========================================================================
d*d data + (d*d - 1) ancilla qubits, (d*d - 1)/2 Z-stabilizers and as many
X-stabilizers, memory-Z protocol (initial |0..0>, final Z-basis data
readout). Everything distance-dependent lives in SurfaceDef; get one with
surface_def(distance). d = 3 (17 qubits) and d = 5 (49 qubits) are built
at import and checked.

Lattice coordinates (x horizontal, y vertical), data on odd-odd sites
1..2d-1, stabilizers on even-even sites 0..2d. d = 3:

    y=5   (1,5)   (3,5)   (5,5)          Z26=(2,6)  boundary (top)
    y=3   (1,3)   (3,3)   (5,3)          X02=(0,2), X64=(6,4) boundary
    y=1   (1,1)   (3,1)   (5,1)          Z40=(4,0)  boundary (bottom)

  A site (x, y) is a Z check when (x + y) / 2 is even, an X check when it
  is odd. Bulk sites (2 <= x, y <= 2d-2) all exist; on the top / bottom
  edges (y = 0, 2d) only the Z sites exist, on the left / right edges
  (x = 0, 2d) only the X sites. d = 3:
    Z: Z22=(2,2), Z44=(4,4) bulk; Z40=(4,0), Z26=(2,6) boundary
    X: X42=(4,2), X24=(2,4) bulk; X02=(0,2), X64=(6,4) boundary
  Check names are "<type><x><y>". CYCLE_ORDER = Z checks then X checks,
  each bulk first then boundary, sorted by (y, x).
  logical Z = data column x=1, logical X = data row y=1.

CX order inside a cycle (hook-error-safe standard order, FIXED constants):
  every stabilizer executes its CXs over 4 shared layers; a stabilizer's
  CX in layer l targets its corner CORNER_ORDER[type][l] (skipped if that
  corner has no data qubit - boundary weight-2 checks keep their slots):
    Z (data->anc): Z_CORNER_ORDER = (-1,-1),(+1,-1),(-1,+1),(+1,+1)
                   ("Z" shape: bottom row left-to-right, then top row)
    X (anc->data): X_CORNER_ORDER = (-1,-1),(-1,+1),(+1,-1),(+1,+1)
                   ("N" shape: left column bottom-to-top, then right)
  Why hook-safe (memory-Z): an X error on an X-ancilla after layer 2
  propagates into the two data of layers 3-4, which under X_CORNER_ORDER
  share the same x - a VERTICAL X pair, parallel to logical Z / adding no
  progress along logical X, so one fault never eats a unit of effective
  distance. Dually, a Z error on a Z-ancilla after layer 2 leaves a
  HORIZONTAL Z pair (same y), benign for logical Z. The schedule is also
  conflict-free: no data qubit is touched twice in one layer (asserted
  for every distance built).
  verification/verify_rotatedSurface.py checks the propagation empirically.

Ancillas are NEVER reset (hardware): raw bit = XOR accumulation with the
same ancilla's previous measurement; check_values() (per-ancilla XOR
chain) recovers the check values. The abstract Stim circuit
(dataset_generation/rotatedSurface_stim.py) uses MR - the two streams
agree at the check-value level, the same contract as heavy-hex.

ibm_miami embedding (45-degree rotation): verified 12x10 row-major
square lattice (q = row*10 + col; 218 undirected edges). The code
lattice maps by u=(x+y)/2, v=(y-x)/2 - diagonal data-ancilla pairs
become unit lattice steps, so EVERY stabilizer CX is device-adjacent and
no SWAP is ever inserted. The patch occupies a (2d-1) x (2d-1) block
placed at EMBED_OFFSETS[backend][d].
"""
import json

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# fixed hook-safe CX corner orders (see module docstring) --------------
Z_CORNER_ORDER = ((-1, -1), (+1, -1), (-1, +1), (+1, +1))
X_CORNER_ORDER = ((-1, -1), (-1, +1), (+1, -1), (+1, +1))
N_LAYERS = 4


def _check_sites(d, ctype):
    """Check coordinates of one type: bulk then boundary, each (y, x)."""
    def is_type(x, y):
        return (((x + y) // 2) % 2 == 0) == (ctype == "Z")

    inner = range(2, 2 * d - 1, 2)
    bulk = [(x, y) for y in inner for x in inner if is_type(x, y)]
    if ctype == "Z":
        edge = [(x, y) for y in (0, 2 * d) for x in inner if is_type(x, y)]
    else:
        edge = [(x, y) for x in (0, 2 * d) for y in inner if is_type(x, y)]
    by_yx = lambda c: (c[1], c[0])                       # noqa: E731
    return sorted(bulk, key=by_yx) + sorted(edge, key=by_yx)


class SurfaceDef:
    """All distance-dependent constants of one rotated surface code.

    DATA_COORDS, DIDX        data coordinates (row-major, y then x) / index
    ANC_DEFS                 name -> (type, coord)
    CYCLE_ORDER              per-cycle measurement order (bit j of a cycle)
    Z_NAMES, X_NAMES, Z_POS  names per type, positions of Z in CYCLE_ORDER
    CHECK_DEFS               name -> (type, support coords, anc coord,
                             ((layer, data coord), ...))
    LOGICAL_Z(_IDX), LOGICAL_X
    GRID_SHAPE, ANC_GRID     (d+1, d+1) plaquette-vertex grid of the CNN
                             tensor; name -> (row = y/2, col = x/2)
    ALL_COORDS, L, AIDX      qubit ordering: data 0..d*d-1, then the
                             ancillas in CYCLE_ORDER; coord -> local index;
                             name -> ancilla local index"""

    def __init__(self, distance):
        d = int(distance)
        if d < 3 or d % 2 == 0:
            raise ValueError(f"surface distance must be odd >= 3, got {d}")
        self.DISTANCE = d
        self.NUM_DATA = d * d
        self.N_CHECKS = d * d - 1
        self.NUM_QUBITS = 2 * d * d - 1
        odd = range(1, 2 * d, 2)
        self.DATA_COORDS = [(x, y) for y in odd for x in odd]
        self.DIDX = {c: i for i, c in enumerate(self.DATA_COORDS)}

        self.ANC_DEFS = {}
        for ctype in ("Z", "X"):
            for (x, y) in _check_sites(d, ctype):
                self.ANC_DEFS[f"{ctype}{x}{y}"] = (ctype, (x, y))
        self.CYCLE_ORDER = list(self.ANC_DEFS)
        assert len(self.CYCLE_ORDER) == self.N_CHECKS
        self.Z_NAMES = [n for n in self.CYCLE_ORDER
                        if self.ANC_DEFS[n][0] == "Z"]
        self.X_NAMES = [n for n in self.CYCLE_ORDER
                        if self.ANC_DEFS[n][0] == "X"]
        self.Z_POS = [j for j, n in enumerate(self.CYCLE_ORDER)
                      if n in self.Z_NAMES]

        self.CHECK_DEFS = {}
        for n in self.CYCLE_ORDER:
            slots = self._layer_slots(n)
            self.CHECK_DEFS[n] = (self.ANC_DEFS[n][0],
                                  tuple(c for _, c in slots),
                                  self.ANC_DEFS[n][1], slots)
        self.Z_STABS = {n: self.CHECK_DEFS[n][1] for n in self.Z_NAMES}
        self.X_STABS = {n: self.CHECK_DEFS[n][1] for n in self.X_NAMES}

        self.LOGICAL_Z = [(1, y) for y in odd]           # one data column
        self.LOGICAL_X = [(x, 1) for x in odd]           # one data row
        self.LOGICAL_Z_IDX = [self.DIDX[c] for c in self.LOGICAL_Z]

        self.GRID_SHAPE = (d + 1, d + 1)
        self.ANC_GRID = {n: (self.ANC_DEFS[n][1][1] // 2,
                             self.ANC_DEFS[n][1][0] // 2)
                         for n in self.CYCLE_ORDER}
        assert len(set(self.ANC_GRID.values())) == self.N_CHECKS

        self.ALL_COORDS = self.DATA_COORDS + [self.ANC_DEFS[n][1]
                                              for n in self.CYCLE_ORDER]
        self.L = {c: i for i, c in enumerate(self.ALL_COORDS)}
        self.AIDX = {n: self.L[self.ANC_DEFS[n][1]]
                     for n in self.CYCLE_ORDER}

        # sanity: schedule is conflict-free (a data qubit once per layer)
        for layer in range(N_LAYERS):
            touched = [c for n in self.CYCLE_ORDER
                       for (ll, c) in self.CHECK_DEFS[n][3] if ll == layer]
            assert len(touched) == len(set(touched)), \
                f"d={d} layer {layer} conflict"
        # sanity: every check has weight 4 (bulk) or 2 (boundary)
        assert all(len(self.CHECK_DEFS[n][1]) in (2, 4)
                   for n in self.CYCLE_ORDER)

    def _layer_slots(self, name):
        """Check -> ((layer, data_coord), ...) honoring the corner orders."""
        ctype, (ax, ay) = self.ANC_DEFS[name]
        order = Z_CORNER_ORDER if ctype == "Z" else X_CORNER_ORDER
        slots = []
        for layer, (dx, dy) in enumerate(order):
            c = (ax + dx, ay + dy)
            if c in self.DIDX:
                slots.append((layer, c))
        return tuple(slots)


_DEFS = {}


def surface_def(distance=3):
    """SurfaceDef of a distance (built once)."""
    distance = int(distance)
    if distance not in _DEFS:
        _DEFS[distance] = SurfaceDef(distance)
    return _DEFS[distance]


for _d in (3, 5):                       # build + self-check at import
    surface_def(_d)


def check_values(syn_rows, num_cycles, distance=3):
    """No-reset raw bits -> per-cycle check values via per-ancilla XOR
    chains (each ancilla measures the same check every cycle)."""
    S = surface_def(distance)
    syn_rows = np.asarray(syn_rows)
    n = syn_rows.shape[0]
    vals = {name: np.zeros((n, num_cycles), dtype=int)
            for name in S.CYCLE_ORDER}
    for j, name in enumerate(S.CYCLE_ORDER):
        prev = np.zeros(n, dtype=int)
        for cyc in range(num_cycles):
            raw = syn_rows[:, cyc * S.N_CHECKS + j]
            vals[name][:, cyc] = raw ^ prev
            prev = raw.copy()
    return vals


# ==================================================================
# hardware circuit (Qiskit) - no-reset ancillas, memory-Z
# ==================================================================
class RotatedSurfaceHardware:
    """Rotated surface code circuit on 2*d*d - 1 qubits (local indices;
    the device embedding only enters at transpile initial_layout).

    inject:      (pauli, data_coord, after_cycle) - data error between
                 cycles (verification)
    inject_ops:  list of (pauli, ("data", coord) | ("anc", name), cycle,
                 after_layer) - error right after a CX layer inside a
                 cycle (hook-error verification)
    """

    def __init__(self, num_cycles=3, distance=3):
        self.num_cycles = num_cycles
        self.distance = distance

    def build_circuit(self, initial_state=0, inject=None, inject_ops=()):
        S = surface_def(self.distance)
        q = QuantumRegister(S.NUM_QUBITS, 'q')
        syn = ClassicalRegister(S.N_CHECKS * self.num_cycles, 'syn')
        dat = ClassicalRegister(S.NUM_DATA, 'data')
        qc = QuantumCircuit(q, syn, dat)
        if initial_state == 1:
            for c in S.LOGICAL_X:
                qc.x(S.L[c])
        for cyc in range(self.num_cycles):
            for n in S.X_NAMES:
                qc.h(S.AIDX[n])
            for layer in range(N_LAYERS):
                for n in S.CYCLE_ORDER:
                    for ll, c in S.CHECK_DEFS[n][3]:
                        if ll != layer:
                            continue
                        if S.CHECK_DEFS[n][0] == 'Z':
                            qc.cx(S.L[c], S.AIDX[n])
                        else:
                            qc.cx(S.AIDX[n], S.L[c])
                for pauli, tgt, icyc, ilayer in inject_ops:
                    if icyc == cyc and ilayer == layer:
                        idx = (S.AIDX[tgt[1]] if tgt[0] == "anc"
                               else S.L[tgt[1]])
                        getattr(qc, pauli.lower())(idx)
            for n in S.X_NAMES:
                qc.h(S.AIDX[n])
            for j, n in enumerate(S.CYCLE_ORDER):      # no reset
                qc.measure(S.AIDX[n], syn[cyc * S.N_CHECKS + j])
            qc.barrier()
            if inject is not None and inject[2] == cyc:
                pauli, dc, _ = inject
                getattr(qc, pauli.lower())(S.L[dc])
                qc.barrier()
        for i, c in enumerate(S.DATA_COORDS):
            qc.measure(S.L[c], dat[i])
        return qc


# ==================================================================
# ibm_miami 45-degree embedding
# ==================================================================
# verified device topology (fetch_coupling): 12 rows x 10 cols,
# row-major numbering q = row*10 + col
MIAMI_ROWS, MIAMI_COLS = 12, 10
# top-left device (row, col) of the (2d-1) x (2d-1) patch block, per
# backend and distance. The d=5 block (9x9) fits at rows 0..3, cols 0..1.
EMBED_OFFSETS = {"ibm_miami": {3: (4, 2), 5: (1, 0)}}


def embedding_for_surface(backend_name, distance=3):
    """coord (x,y) -> device qubit, via u=(x+y)/2, v=(y-x)/2 (+ offset).

    Diagonal data-ancilla pairs map to unit lattice steps, so every
    stabilizer CX is device-adjacent (no SWAPs)."""
    S = surface_def(distance)
    try:
        r0, c0 = EMBED_OFFSETS[backend_name][S.DISTANCE]
    except KeyError:
        raise RuntimeError(
            f"no rotatedSurface d={S.DISTANCE} embedding registered for "
            f"backend '{backend_name}'. Known: "
            f"{ {b: sorted(v) for b, v in EMBED_OFFSETS.items()} }. Add a "
            f"block offset to EMBED_OFFSETS in "
            f"circuits/rotatedSurface/rotatedSurface.py after checking "
            f"the device is a row-major square lattice "
            f"(validate_backend_surface).")
    emb = {}
    for (x, y) in S.ALL_COORDS:
        u, v = (x + y) // 2, (y - x) // 2
        row, col = v + (S.DISTANCE - 1) + r0, u - 1 + c0
        emb[(x, y)] = row * MIAMI_COLS + col
    return emb


def required_edges_surface(distance=3):
    """All (coord, coord) pairs a stabilizer CX needs (undirected)."""
    S = surface_def(distance)
    need = []
    for n in S.CYCLE_ORDER:
        anc = S.ANC_DEFS[n][1]
        for c in S.CHECK_DEFS[n][1]:
            need.append((anc, c))
    return need


def validate_backend_surface(coupling_json_path, raise_on_fail=True,
                             distance=3):
    """Verify the patch fits the backend under its embedding.

    Also re-checks the square-lattice assumption on the patch block: if
    the device is NOT the expected row-major square lattice, this fails -
    stop and report to the user instead of forcing the embedding."""
    cm = json.load(open(coupling_json_path))
    emb = embedding_for_surface(cm.get('name', '?'), distance)
    edges = {tuple(sorted(e)) for e in cm['coupling_map']}
    missing = []
    for u, v in required_edges_surface(distance):
        dev = tuple(sorted((emb[u], emb[v])))
        if dev not in edges:
            missing.append(((u, v), dev))
    if missing and raise_on_fail:
        raise RuntimeError(
            f"backend '{cm.get('name', '?')}' lacks {len(missing)} device "
            f"edges required by the rotatedSurface d={distance} 45-degree "
            f"embedding (coupling map differs from the expected square "
            f"lattice - report to the user): {missing}")
    return missing


if __name__ == '__main__':
    for d in (3, 5):
        S = surface_def(d)
        qc = RotatedSurfaceHardware(2, d).build_circuit()
        ops = qc.count_ops()
        print(f"d={d} 2-cycle: qubits={qc.num_qubits} depth={qc.depth()} "
              f"cx={ops.get('cx')} h={ops.get('h')} "
              f"meas={ops.get('measure')} reset={ops.get('reset', 0)}")
        print("  checks:", S.CYCLE_ORDER)
        emb = embedding_for_surface('ibm_miami', d)
        print(f"  miami qubits {min(emb.values())}..{max(emb.values())}")
