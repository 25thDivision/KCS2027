#!/usr/bin/env python3
"""
Triangular color code - abstract Stim circuit generator (d = 3, 5)
==================================================================
Same conventions as the heavy-hex and surface generators; every function
takes distance (default 3) and reads the code constants from
circuits.color.colorcode.color_def(distance):

  * per-cycle measurement order = ColorDef.CYCLE_ORDER (Z0 .. Z{F-1},
    X0 .. X{F-1}; syn bit index = cycle*N_CHECKS + j) - identical to the
    hardware circuit's clbit layout
  * one ancilla per check, MR every cycle; a syn bit is the check value
    (the hardware uses fresh or reset ancillas, colorcode.check_values)
  * inside a cycle the checks run one after another in CYCLE_ORDER, each
    coupling its ancilla to the face qubits in ascending data index -
    the SAME CX order as the hardware circuit, so the spread of an
    ancilla fault into the face is identical
  * detectors: Z-checks from cycle 0 (anchored on the deterministic 0 of
    |0..0>), X-checks from cycle 1 (temporal XOR), then one final-Z
    detector per face (final-data face parity ^ last Z-check value);
    observable = logical Z
  * labels: final-data MEASUREMENT FLIPS via stim.FlipSimulator
  * noise model: the same 4-parameter profiles (data_depol / meas_flip /
    reset_flip / gate_depol) applied at the same kinds of positions as
    the surface generator

Tensor: (shots, 2*num_cycles, H, W) on ColorDef.GRID_SHAPE - the Z check
and the X check of a face share the cell (ANC_GRID); channels [Z-plane,
X-plane] per cycle like the other codes.
"""
import sys
from pathlib import Path

import numpy as np
import stim

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from circuits.color.colorcode import color_def  # noqa: E402
from dataset_generation.heavyhex33_stim import (  # noqa: E402
    NOISE_PROFILES, is_qpu_profile)


def num_detectors(num_cycles, distance=3):
    """Z: F*cycles, X: F*(cycles-1), final: F"""
    f = color_def(distance).N_FACES
    return f * num_cycles + f * (num_cycles - 1) + f


def build_color_stim_circuit(num_cycles=3, noise_type="X", p=0.0,
                             noise_profile="ideal/dp0_mf0_rf0_gd0",
                             inject=None, distance=3):
    """Abstract color-code Stim circuit. Only the initial state |0>_L.

    Args mirror heavyhex33_stim.build_stim_circuit; inject = (pauli,
    data index, after_cycle) as in colorcode.ColorCodeHardware."""
    if is_qpu_profile(noise_profile):
        raise ValueError("qpu_avg profiles are not defined for the color "
                         "code")
    S = color_def(distance)
    prof = (NOISE_PROFILES[noise_profile] if isinstance(noise_profile, str)
            else dict(noise_profile))
    dp, mf = prof["data_depol"], prof["meas_flip"]
    rf, gd = prof["reset_flip"], prof["gate_depol"]

    n = S.NUM_DATA
    data = list(range(n))
    aidx = {name: n + j for j, name in enumerate(S.CYCLE_ORDER)}
    ancs = [aidx[name] for name in S.CYCLE_ORDER]
    c = stim.Circuit()

    c.append("R", data + ancs)
    if rf > 0:
        c.append("X_ERROR", data + ancs, rf)
    if p > 0:
        c.append("X_ERROR" if noise_type == "X" else "Z_ERROR", data, p)

    for cyc in range(num_cycles):
        if dp > 0:
            c.append("DEPOLARIZE1", data, dp)
        for name in S.CYCLE_ORDER:
            ctype, support = S.CHECK_DEFS[name][0], S.CHECK_DEFS[name][1]
            a = aidx[name]
            if ctype == "X":
                c.append("H", [a])
                if gd > 0:
                    c.append("DEPOLARIZE1", [a], gd)
            for i in support:
                pair = [i, a] if ctype == "Z" else [a, i]
                c.append("CX", pair)
                if gd > 0:
                    c.append("DEPOLARIZE2", pair, gd)
            if ctype == "X":
                c.append("H", [a])
                if gd > 0:
                    c.append("DEPOLARIZE1", [a], gd)
            if mf > 0:
                c.append("X_ERROR", [a], mf)
            c.append("MR", [a])
            if rf > 0:
                c.append("X_ERROR", [a], rf)
        if inject is not None and inject[2] == cyc:
            pauli, dq, _ = inject
            c.append(pauli.upper(), [dq])

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
            if name in S.Z_STABS:
                if cyc == 0:
                    c.append("DETECTOR", [rec(syn(0, j))])
                else:
                    c.append("DETECTOR",
                             [rec(syn(cyc, j)), rec(syn(cyc - 1, j))])
            elif cyc >= 1:
                c.append("DETECTOR", [rec(syn(cyc, j)), rec(syn(cyc - 1, j))])
    for j in S.Z_POS:
        name = S.CYCLE_ORDER[j]
        targets = [rec(fin(i)) for i in S.CHECK_DEFS[name][1]]
        targets.append(rec(syn(num_cycles - 1, j)))
        c.append("DETECTOR", targets)
    c.append("OBSERVABLE_INCLUDE", [rec(fin(i)) for i in S.LOGICAL_Z_IDX], 0)


# ==================================================================
# sampling / tensor / detector utilities (mirror the other codes)
# ==================================================================
def split_color_sample(raw, num_cycles, distance=3):
    S = color_def(distance)
    raw = np.asarray(raw, dtype=np.uint8)
    return raw[:, :S.N_CHECKS * num_cycles], raw[:, S.N_CHECKS * num_cycles:]


def sample_flips_color(circuit, shots, num_cycles, seed=None, distance=3):
    """FlipSimulator measurement flips -> (syn check-value flips, data
    flips). Stim uses MR, so syn flips ARE check-value flips."""
    fs = stim.FlipSimulator(batch_size=shots, seed=seed,
                            disable_stabilizer_randomization=True)
    fs.do(circuit)
    mf = fs.get_measurement_flips().T.astype(np.uint8)
    return split_color_sample(mf, num_cycles, distance)


def syndrome_tensor_color(check_mat, num_cycles, distance=3):
    """(shots, N_CHECKS*C) check values -> (shots, 2C, H, W) tensor,
    channels [Z-plane (value), X-plane (cycle-to-cycle XOR; c=0 stays 0)]."""
    S = color_def(distance)
    shots = check_mat.shape[0]
    t = np.zeros((shots, 2 * num_cycles, *S.GRID_SHAPE), dtype=np.uint8)
    for j, name in enumerate(S.CYCLE_ORDER):
        r, col = S.ANC_GRID[name]
        for cyc in range(num_cycles):
            v = check_mat[:, cyc * S.N_CHECKS + j]
            if name in S.Z_STABS:
                t[:, 2 * cyc, r, col] = v
            elif cyc >= 1:
                prev = check_mat[:, (cyc - 1) * S.N_CHECKS + j]
                t[:, 2 * cyc + 1, r, col] = v ^ prev
    return t


def check_matrix_from_dict_color(vals, num_cycles, distance=3):
    S = color_def(distance)
    n = vals[S.CYCLE_ORDER[0]].shape[0]
    mat = np.zeros((n, S.N_CHECKS * num_cycles), dtype=np.uint8)
    for j, name in enumerate(S.CYCLE_ORDER):
        for cyc in range(num_cycles):
            mat[:, cyc * S.N_CHECKS + j] = vals[name][:, cyc]
    return mat


def detectors_from_dataset_color(check_mat, y_qubit, num_cycles, distance=3):
    """(check values, final data bits) -> detector vectors in
    _append_detectors order (hardware path)."""
    S = color_def(distance)
    shots = check_mat.shape[0]
    cols = []
    for cyc in range(num_cycles):
        for j, name in enumerate(S.CYCLE_ORDER):
            v = check_mat[:, cyc * S.N_CHECKS + j]
            if name in S.Z_STABS:
                if cyc == 0:
                    cols.append(v)
                else:
                    cols.append(v ^ check_mat[:, (cyc - 1) * S.N_CHECKS + j])
            elif cyc >= 1:
                cols.append(v ^ check_mat[:, (cyc - 1) * S.N_CHECKS + j])
    for j in S.Z_POS:
        name = S.CYCLE_ORDER[j]
        par = np.zeros(shots, dtype=np.uint8)
        for i in S.CHECK_DEFS[name][1]:
            par ^= y_qubit[:, i]
        cols.append(par ^ check_mat[:, (num_cycles - 1) * S.N_CHECKS + j])
    det = np.stack(cols, axis=1).astype(np.uint8)
    assert det.shape[1] == num_detectors(num_cycles, distance)
    return det


def detectors_from_tensor_color(tensor, y_qubit, distance=3):
    """Saved dataset (features, labels) -> detector vectors."""
    S = color_def(distance)
    tensor = np.asarray(tensor, dtype=np.uint8)
    y_qubit = np.asarray(y_qubit, dtype=np.uint8)
    shots, ch = tensor.shape[0], tensor.shape[1]
    assert ch % 2 == 0
    assert tensor.shape[2:] == S.GRID_SHAPE, \
        f"tensor grid {tensor.shape[2:]} != color d={distance} grid"
    num_cycles = ch // 2
    cols = []
    for cyc in range(num_cycles):
        for name in S.CYCLE_ORDER:
            r, col = S.ANC_GRID[name]
            if name in S.Z_STABS:
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
        for i in S.CHECK_DEFS[name][1]:
            par ^= y_qubit[:, i]
        cols.append(par ^ tensor[:, 2 * (num_cycles - 1), r, col])
    det = np.stack(cols, axis=1).astype(np.uint8)
    assert det.shape[1] == num_detectors(num_cycles, distance)
    return det


def logical_label_color(y_qubit, distance=3):
    """Final data bits -> logical Z flip (LOGICAL_Z parity)."""
    S = color_def(distance)
    lab = np.zeros(y_qubit.shape[0], dtype=np.uint8)
    for i in S.LOGICAL_Z_IDX:
        lab ^= y_qubit[:, i]
    return lab


if __name__ == "__main__":
    for d in (3, 5):
        c = build_color_stim_circuit(
            3, "X", 0.01, "realistic/dp0.001_mf0.01_rf0.01_gd0.008",
            distance=d)
        print(f"d={d} cycles=3 qubits={c.num_qubits} "
              f"meas={c.num_measurements} detectors={c.num_detectors} "
              f"(expected {num_detectors(3, d)})")
        c0 = build_color_stim_circuit(3, distance=d)
        det = c0.compile_detector_sampler().sample(shots=256,
                                                   append_observables=True)
        print(f"  noiseless detectors all-zero: {np.asarray(det).sum() == 0}")
