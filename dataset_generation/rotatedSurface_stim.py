#!/usr/bin/env python3
"""
Rotated surface code - abstract Stim circuit generator (any odd distance)
=========================================================================
Same conventions as the heavy-hex generator (heavyhex33_stim.py); every
function takes distance (default 3) and reads the code constants from
circuits.rotatedSurface.rotatedSurface.surface_def(distance):

  * per-cycle measurement order = SurfaceDef.CYCLE_ORDER (N_CHECKS =
    d*d - 1 checks; syn bit index = cycle*N_CHECKS + j) - identical to
    the hardware circuit's clbit layout
  * Stim treats every ancilla with MR; the hardware never resets, and
    rotatedSurface.check_values() (per-ancilla XOR chain) recovers the
    check values - the two streams agree at the CHECK-VALUE level (same
    contract as heavy-hex)
  * CX schedule inside a cycle = the SAME fixed hook-safe 4-layer order
    as the hardware circuit (Z_CORNER_ORDER / X_CORNER_ORDER), so
    hook-error propagation is identical between Stim and Aer
    (verification/verify_rotatedSurface.py checks this)
  * detectors: Z-checks from cycle 0 (anchored on the deterministic 0 of
    |0>_L), X-checks from cycle 1 (temporal XOR), then the final-Z
    detectors (final-data support parity ^ last Z-check value);
    observable = logical Z = parity of the data column x=1
  * labels: final-data MEASUREMENT FLIPS via stim.FlipSimulator (the
    X-checks project |0>_L, so individual final bits are random and only
    parities are deterministic - same subtlety as heavy-hex)
  * noise model: identical 4-parameter profiles (data_depol / meas_flip /
    reset_flip / gate_depol) applied at the same circuit positions

Tensor: (shots, 2*num_cycles, d+1, d+1) - the ancillas on the
plaquette-vertex grid (SurfaceDef.ANC_GRID), channels [Z-plane, X-plane]
per cycle exactly like the heavy-hex diamond tensor.
"""
import sys
from pathlib import Path

import numpy as np
import stim

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from circuits.rotatedSurface.rotatedSurface import (  # noqa: E402
    surface_def, N_LAYERS)
from dataset_generation.heavyhex33_stim import (  # noqa: E402
    NOISE_PROFILES, is_qpu_profile)


def num_detectors(num_cycles, distance=3):
    """Z: nz*cycles, X: nx*(cycles-1), final: nz"""
    S = surface_def(distance)
    nz, nx = len(S.Z_NAMES), len(S.X_NAMES)
    return nz * num_cycles + nx * (num_cycles - 1) + nz


def build_rotatedSurface_stim_circuit(num_cycles=3, noise_type="X", p=0.0,
                                      noise_profile="ideal/dp0_mf0_rf0_gd0",
                                      inject=None, inject_ops=(),
                                      distance=3):
    """Abstract rotated-surface Stim circuit. Only the initial state |0>_L.

    Args mirror heavyhex33_stim.build_stim_circuit; additionally
    inject_ops = list of (pauli, ("data", coord)|("anc", name), cycle,
    after_layer) for hook-error verification (same meaning as
    rotatedSurface.RotatedSurfaceHardware.build_circuit)."""
    if is_qpu_profile(noise_profile):
        raise ValueError("qpu_avg profiles need the hardware-shaped "
                         "circuit (rotatedSurface_qpu_stim)")
    S = surface_def(distance)
    prof = (NOISE_PROFILES[noise_profile] if isinstance(noise_profile, str)
            else dict(noise_profile))
    dp, mf = prof["data_depol"], prof["meas_flip"]
    rf, gd = prof["reset_flip"], prof["gate_depol"]

    data = list(range(S.NUM_DATA))
    ancs = [S.AIDX[n] for n in S.CYCLE_ORDER]
    xancs = [S.AIDX[n] for n in S.X_NAMES]
    c = stim.Circuit()

    c.append("R", data + ancs)
    if rf > 0:
        c.append("X_ERROR", data + ancs, rf)
    if p > 0:
        c.append("X_ERROR" if noise_type == "X" else "Z_ERROR", data, p)

    for cyc in range(num_cycles):
        if dp > 0:
            c.append("DEPOLARIZE1", data, dp)
        c.append("H", xancs)
        if gd > 0:
            c.append("DEPOLARIZE1", xancs, gd)
        for layer in range(N_LAYERS):
            for name in S.CYCLE_ORDER:
                ctype = S.CHECK_DEFS[name][0]
                for ll, coord in S.CHECK_DEFS[name][3]:
                    if ll != layer:
                        continue
                    pair = ([S.DIDX[coord], S.AIDX[name]] if ctype == "Z"
                            else [S.AIDX[name], S.DIDX[coord]])
                    c.append("CX", pair)
                    if gd > 0:
                        c.append("DEPOLARIZE2", pair, gd)
            for pauli, tgt, icyc, ilayer in inject_ops:
                if icyc == cyc and ilayer == layer:
                    idx = (S.AIDX[tgt[1]] if tgt[0] == "anc"
                           else S.DIDX[tgt[1]])
                    c.append(pauli.upper(), [idx])
        c.append("H", xancs)
        if gd > 0:
            c.append("DEPOLARIZE1", xancs, gd)
        for name in S.CYCLE_ORDER:                     # MR in bit order
            a = S.AIDX[name]
            if mf > 0:
                c.append("X_ERROR", [a], mf)
            c.append("MR", [a])
            if rf > 0:
                c.append("X_ERROR", [a], rf)
        if inject is not None and inject[2] == cyc:
            pauli, dc, _ = inject
            c.append(pauli.upper(), [S.DIDX[dc]])

    if mf > 0:
        c.append("X_ERROR", data, mf)
    c.append("M", data)
    _append_detectors(c, num_cycles, S)
    return c


def _append_detectors(c, num_cycles, S):
    """Detector order (must match detectors_from_tensor/-dataset below):
      1) per cycle, iterate CYCLE_ORDER: Z-checks every cycle (cycle 0
         anchored on the deterministic 0), X-checks from cycle 1
      2) the final-Z detectors (Z_POS order)"""
    M = S.N_CHECKS * num_cycles + S.NUM_DATA
    rec = lambda k: stim.target_rec(k - M)  # noqa: E731
    syn = lambda cyc, j: cyc * S.N_CHECKS + j  # noqa: E731
    fin = lambda i: S.N_CHECKS * num_cycles + i  # noqa: E731

    for cyc in range(num_cycles):
        for j, name in enumerate(S.CYCLE_ORDER):
            if name in S.Z_NAMES:
                if cyc == 0:
                    c.append("DETECTOR", [rec(syn(0, j))])
                else:
                    c.append("DETECTOR",
                             [rec(syn(cyc, j)), rec(syn(cyc - 1, j))])
            elif cyc >= 1:
                c.append("DETECTOR", [rec(syn(cyc, j)), rec(syn(cyc - 1, j))])
    for j in S.Z_POS:
        name = S.CYCLE_ORDER[j]
        targets = [rec(fin(S.DIDX[qc])) for qc in S.CHECK_DEFS[name][1]]
        targets.append(rec(syn(num_cycles - 1, j)))
        c.append("DETECTOR", targets)
    c.append("OBSERVABLE_INCLUDE", [rec(fin(i)) for i in S.LOGICAL_Z_IDX], 0)


# ==================================================================
# sampling / tensor / detector utilities (mirror heavyhex33_stim)
# ==================================================================
def split_rotatedSurface_sample(raw, num_cycles, distance=3):
    S = surface_def(distance)
    raw = np.asarray(raw, dtype=np.uint8)
    return raw[:, :S.N_CHECKS * num_cycles], raw[:, S.N_CHECKS * num_cycles:]


def sample_flips_rotatedSurface(circuit, shots, num_cycles, seed=None,
                                distance=3):
    """FlipSimulator measurement flips -> (syn check-value flips, data
    flips). Stim uses MR, so syn flips ARE check-value flips."""
    fs = stim.FlipSimulator(batch_size=shots, seed=seed,
                            disable_stabilizer_randomization=True)
    fs.do(circuit)
    mf = fs.get_measurement_flips().T.astype(np.uint8)
    return split_rotatedSurface_sample(mf, num_cycles, distance)


def syndrome_tensor_rotatedSurface(check_mat, num_cycles, distance=3):
    """(shots, N_CHECKS*C) check values -> (shots, 2C, d+1, d+1) tensor,
    channels [Z-plane (value), X-plane (cycle-to-cycle XOR; c=0 stays 0)]."""
    S = surface_def(distance)
    shots = check_mat.shape[0]
    t = np.zeros((shots, 2 * num_cycles, *S.GRID_SHAPE), dtype=np.uint8)
    for j, name in enumerate(S.CYCLE_ORDER):
        r, col = S.ANC_GRID[name]
        for cyc in range(num_cycles):
            v = check_mat[:, cyc * S.N_CHECKS + j]
            if name in S.Z_NAMES:
                t[:, 2 * cyc, r, col] = v
            elif cyc >= 1:
                prev = check_mat[:, (cyc - 1) * S.N_CHECKS + j]
                t[:, 2 * cyc + 1, r, col] = v ^ prev
    return t


def check_matrix_from_dict_rotatedSurface(vals, num_cycles, distance=3):
    S = surface_def(distance)
    n = vals[S.CYCLE_ORDER[0]].shape[0]
    mat = np.zeros((n, S.N_CHECKS * num_cycles), dtype=np.uint8)
    for j, name in enumerate(S.CYCLE_ORDER):
        for cyc in range(num_cycles):
            mat[:, cyc * S.N_CHECKS + j] = vals[name][:, cyc]
    return mat


def detectors_from_dataset_rotatedSurface(check_mat, y_qubit, num_cycles,
                                          distance=3):
    """(check values, final data bits) -> detector vectors in
    _append_detectors order (hardware/MWPM path)."""
    S = surface_def(distance)
    shots = check_mat.shape[0]
    cols = []
    for cyc in range(num_cycles):
        for j, name in enumerate(S.CYCLE_ORDER):
            v = check_mat[:, cyc * S.N_CHECKS + j]
            if name in S.Z_NAMES:
                if cyc == 0:
                    cols.append(v)
                else:
                    cols.append(v ^ check_mat[:, (cyc - 1) * S.N_CHECKS + j])
            elif cyc >= 1:
                cols.append(v ^ check_mat[:, (cyc - 1) * S.N_CHECKS + j])
    for j in S.Z_POS:
        name = S.CYCLE_ORDER[j]
        par = np.zeros(shots, dtype=np.uint8)
        for qc in S.CHECK_DEFS[name][1]:
            par ^= y_qubit[:, S.DIDX[qc]]
        cols.append(par ^ check_mat[:, (num_cycles - 1) * S.N_CHECKS + j])
    det = np.stack(cols, axis=1).astype(np.uint8)
    assert det.shape[1] == num_detectors(num_cycles, distance)
    return det


def detectors_from_tensor_rotatedSurface(tensor, y_qubit, distance=3):
    """Saved dataset (features, labels) -> detector vectors (MWPM path)."""
    S = surface_def(distance)
    tensor = np.asarray(tensor, dtype=np.uint8)
    y_qubit = np.asarray(y_qubit, dtype=np.uint8)
    shots, ch = tensor.shape[0], tensor.shape[1]
    assert ch % 2 == 0
    assert tensor.shape[2:] == S.GRID_SHAPE, \
        f"tensor grid {tensor.shape[2:]} != surface d={distance} grid"
    num_cycles = ch // 2
    cols = []
    for cyc in range(num_cycles):
        for name in S.CYCLE_ORDER:
            r, col = S.ANC_GRID[name]
            if name in S.Z_NAMES:
                if cyc == 0:
                    cols.append(tensor[:, 0, r, col])
                else:
                    cols.append(tensor[:, 2 * cyc, r, col]
                                ^ tensor[:, 2 * (cyc - 1), r, col])
            elif cyc >= 1:
                cols.append(tensor[:, 2 * cyc + 1, r, col])
    for j in S.Z_POS:
        name = S.CYCLE_ORDER[j]
        r, col = S.ANC_GRID[name]
        par = np.zeros(shots, dtype=np.uint8)
        for qc in S.CHECK_DEFS[name][1]:
            par ^= y_qubit[:, S.DIDX[qc]]
        cols.append(par ^ tensor[:, 2 * (num_cycles - 1), r, col])
    det = np.stack(cols, axis=1).astype(np.uint8)
    assert det.shape[1] == num_detectors(num_cycles, distance)
    return det


def logical_label_rotatedSurface(y_qubit, distance=3):
    """Final data bits -> logical Z flip (column x=1 parity)."""
    S = surface_def(distance)
    lab = np.zeros(y_qubit.shape[0], dtype=np.uint8)
    for i in S.LOGICAL_Z_IDX:
        lab ^= y_qubit[:, i]
    return lab


if __name__ == "__main__":
    for d in (3, 5):
        c = build_rotatedSurface_stim_circuit(
            3, "X", 0.01, "realistic/dp0.001_mf0.01_rf0.01_gd0.008",
            distance=d)
        print(f"d={d} cycles=3 qubits={c.num_qubits} "
              f"meas={c.num_measurements} detectors={c.num_detectors} "
              f"(expected {num_detectors(3, d)})")
        dem = c.detector_error_model(decompose_errors=True)
        print(f"  DEM instructions: {len(dem)}")
        c0 = build_rotatedSurface_stim_circuit(3, distance=d)
        det = c0.compile_detector_sampler().sample(shots=256,
                                                   append_observables=True)
        print(f"  noiseless detectors all-zero: {np.asarray(det).sum() == 0}")
