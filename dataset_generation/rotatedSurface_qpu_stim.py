#!/usr/bin/env python3
"""
Hardware-shaped rotated surface Stim circuit (any odd distance) with
QPU-averaged calibration noise
============================================================
The surface counterpart of heavyhex37_qpu_stim.py. Mirrors
circuits.rotatedSurface.rotatedSurface.RotatedSurfaceHardware gate-for-gate
on the 2*d*d - 1 patch qubits (d*d data + d*d - 1 ancillas, local indices
in ALL_COORDS order; d = 3: 17 qubits, data 0-8, ancillas 9-16): the same fixed
hook-safe 4-layer CX schedule (Z_CORNER_ORDER / X_CORNER_ORDER) and NO
ancilla reset - raw ancilla bits accumulate exactly like on hardware and
check values are recovered by the per-ancilla XOR chain
(rotatedSurface.check_values).

Noise attachment (profile mode "qpu_avg_v1" with code "surface", built
by make_qpu_avg_profile.py --code surface; every rate is the
run-averaged calibration value of THAT patch qubit / edge, keyed by the
patch LOCAL index (data first, then the ancillas in CYCLE_ORDER) and
"i-j" for edges):
  * each CX               -> DEPOLARIZE2(edge 2Q error) on its qubits
  * each H                -> DEPOLARIZE1(qubit 1Q error, sx proxy)
  * right before each M   -> X_ERROR(qubit readout error)
  * injected data error   -> X_ERROR/Z_ERROR(p) on data right after the
                             initial reset (same Error_Type/Error_Rate
                             axis as the abstract datasets)

NOT modeled (documented limitation, also in README.md):
  * calibration drift between/within runs (values are averages)
  * T1/T2 idle decoherence and delay errors (incl. DD behavior)
  * measurement crosstalk / correlated readout errors
  * coherent (non-Pauli) errors; the transpiled 1Q-gate structure is
    approximated by one sx-proxy depolarizing event per H
  * ibm_miami is CZ-native: the calibrated 2Q error is a CZ benchmark
    value attached at the CX position - a proxy, exactly like the ECR
    proxy of the heavy-hex generator

Detectors are defined on the RAW measurement record via XOR expansion of
the per-ancilla chains (raw_j ^ raw_{j-1} = check value), in EXACTLY the
detector order of rotatedSurface_stim._append_detectors - so the tensor / MWPM
reconstruction logic is unchanged and the DEM can feed PyMatching.

verification/verify_rotatedSurface.py section [G] enforces detector determinism,
single-error signature agreement vs Aer, and check-value-level
equivalence with the abstract surface Stim circuit; qpu/* surface dataset
generation must NOT proceed before ALL PASS (incl. [G]).
"""
import sys
from pathlib import Path

import numpy as np
import stim

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from circuits.rotatedSurface.rotatedSurface import (  # noqa: E402
    surface_def, N_LAYERS, check_values)
from dataset_generation.rotatedSurface_stim import (  # noqa: E402
    num_detectors, split_rotatedSurface_sample,
    check_matrix_from_dict_rotatedSurface)

_ZERO = {"readout": {}, "error_1q": {}, "error_2q": {}}

# stim/DEM-safe probability caps (defense-in-depth: the profile
# generator already clamps dead values, but a hand-edited or legacy
# profile must never crash the sampler / detector-error-model)
_CAP = {"readout": 0.5, "error_1q": 0.5, "error_2q": 0.75}


def _capped(d, kind):
    out, clamped = {}, []
    for k, v in d.items():
        if v is None or v > _CAP[kind]:
            clamped.append((k, v))
            v = _CAP[kind]
        out[k] = v
    if clamped:
        print(f"WARNING: {kind} {len(clamped)} value(s) clamped to "
              f"{_CAP[kind]} (dead/over-limit calibration): "
              f"{clamped[:4]}{'...' if len(clamped) > 4 else ''}")
    return out


def _rates(profile):
    """Profile dict -> lookup helpers over LOCAL indices (0.0 where
    absent / noiseless). Rejects heavyhex-tagged profiles."""
    prof = profile or _ZERO
    if profile is not None and prof.get("code", "heavyhex") != "surface":
        raise ValueError(
            "rotatedSurface_qpu_stim needs a --code surface qpu_avg profile "
            f"(got code '{prof.get('code', 'heavyhex')}')")
    ro = _capped({int(k): v for k, v in prof.get("readout", {}).items()},
                 "readout")
    e1 = _capped({int(k): v for k, v in prof.get("error_1q", {}).items()},
                 "error_1q")
    e2 = {}
    for k, v in prof.get("error_2q", {}).items():
        u, w = (int(x) for x in k.split("-"))
        e2[tuple(sorted((u, w)))] = v
    e2 = _capped(e2, "error_2q")
    return (lambda q: ro.get(q, 0.0),
            lambda q: e1.get(q, 0.0),
            lambda a, b: e2.get(tuple(sorted((a, b))), 0.0))


class _Builder:
    def __init__(self, profile):
        self.c = stim.Circuit()
        self.read_err, self.h_err, self.cx_err = _rates(profile)
        self.n_meas = 0
        self.prev_meas = {}         # anc local idx -> previous meas index
        self.chain = {}             # (name, cyc) -> [raw indices] whose
        #                             XOR is the check VALUE

    def cx(self, a, b):             # local indices
        self.c.append("CX", [a, b])
        p = self.cx_err(a, b)
        if p > 0:
            self.c.append("DEPOLARIZE2", [a, b], p)

    def h(self, q):
        self.c.append("H", [q])
        p = self.h_err(q)
        if p > 0:
            self.c.append("DEPOLARIZE1", [q], p)

    def measure(self, q):
        p = self.read_err(q)
        if p > 0:
            self.c.append("X_ERROR", [q], p)
        self.c.append("M", [q])          # no reset
        self.n_meas += 1
        return self.n_meas - 1


def build_rotatedSurface_qpu_stim_circuit(num_cycles=3, noise_type="X", p=0.0,
                                          profile=None, inject=None,
                                          distance=3):
    """Hardware-shaped surface Stim circuit (initial state |0>_L only).

    Args mirror heavyhex37_qpu_stim.build_qpu_stim_circuit:
        profile: surface qpu_avg_v1 profile dict (None -> noiseless,
                 used by the verification gate)
        inject:  (pauli, data_coord, after_cycle) - deterministic single
                 error for verification (same meaning as
                 rotatedSurface.RotatedSurfaceHardware.build_circuit)
    """
    S = surface_def(distance)
    if profile is not None and int(profile.get("distance", 3)) != S.DISTANCE:
        raise ValueError(
            f"qpu_avg profile is for surface d={profile.get('distance', 3)}"
            f", not d={S.DISTANCE}")
    b = _Builder(profile)
    b.c.append("R", list(range(len(S.ALL_COORDS))))
    if p > 0:
        b.c.append("X_ERROR" if noise_type == "X" else "Z_ERROR",
                   list(range(S.NUM_DATA)), p)

    for cyc in range(num_cycles):
        for n in S.X_NAMES:
            b.h(S.AIDX[n])
        for layer in range(N_LAYERS):
            for name in S.CYCLE_ORDER:
                ctype = S.CHECK_DEFS[name][0]
                for ll, coord in S.CHECK_DEFS[name][3]:
                    if ll != layer:
                        continue
                    if ctype == "Z":
                        b.cx(S.DIDX[coord], S.AIDX[name])
                    else:
                        b.cx(S.AIDX[name], S.DIDX[coord])
        for n in S.X_NAMES:
            b.h(S.AIDX[n])
        for name in S.CYCLE_ORDER:             # measure in bit order
            a = S.AIDX[name]
            m = b.measure(a)
            prev = b.prev_meas.get(a)
            b.chain[(name, cyc)] = [m] if prev is None else [m, prev]
            b.prev_meas[a] = m
        if inject is not None and inject[2] == cyc:
            pauli, dc, _ = inject
            b.c.append(pauli.upper(), [S.DIDX[dc]])

    fin = {}
    for c in S.DATA_COORDS:                    # final data readout
        fin[c] = b.measure(S.L[c])

    # ---- detectors: XOR expansion of the raw record, in the exact
    # ---- order of rotatedSurface_stim._append_detectors ------------------
    M = b.n_meas
    assert M == S.N_CHECKS * num_cycles + S.NUM_DATA
    rec = lambda k: stim.target_rec(k - M)  # noqa: E731

    def value_set(name, cyc):
        return set(b.chain[(name, cyc)])

    for cyc in range(num_cycles):
        for name in S.CYCLE_ORDER:
            if name in S.Z_NAMES:
                s = value_set(name, cyc)
                if cyc >= 1:
                    s = s ^ value_set(name, cyc - 1)
                b.c.append("DETECTOR", [rec(k) for k in sorted(s)])
            elif cyc >= 1:
                s = value_set(name, cyc) ^ value_set(name, cyc - 1)
                b.c.append("DETECTOR", [rec(k) for k in sorted(s)])
    for j in S.Z_POS:
        name = S.CYCLE_ORDER[j]
        s = {fin[qc] for qc in S.CHECK_DEFS[name][1]}
        s ^= value_set(name, num_cycles - 1)
        b.c.append("DETECTOR", [rec(k) for k in sorted(s)])
    b.c.append("OBSERVABLE_INCLUDE",
               [rec(fin[S.DATA_COORDS[i]]) for i in S.LOGICAL_Z_IDX], 0)
    assert b.c.num_detectors == num_detectors(num_cycles, distance)
    return b.c


def sample_rotatedSurface_qpu_flips(circuit, shots, num_cycles, seed=None,
                                    distance=3):
    """FlipSimulator flips of the RAW record -> (check-value flip matrix
    (shots, N_CHECKS*C), final-data flips (shots, NUM_DATA)).

    The raw syn flips pass through the same per-ancilla XOR chain as
    hardware raw bits (rotatedSurface.check_values), which is linear, so the
    result is the flip of each CHECK VALUE - directly comparable to the
    abstract circuit's MR bits and consumable by syndrome_tensor_rotatedSurface /
    the detector reconstructions."""
    fs = stim.FlipSimulator(batch_size=shots, seed=seed,
                            disable_stabilizer_randomization=True)
    fs.do(circuit)
    mf = fs.get_measurement_flips().T.astype(np.uint8)
    raw_syn, dat = split_rotatedSurface_sample(mf, num_cycles, distance)
    vals = check_values(raw_syn, num_cycles, distance)
    return check_matrix_from_dict_rotatedSurface(vals, num_cycles,
                                                 distance), dat


if __name__ == "__main__":
    for d in (3, 5):
        c = build_rotatedSurface_qpu_stim_circuit(3, distance=d)
        print(f"d={d} noiseless: qubits={c.num_qubits} "
              f"meas={c.num_measurements} detectors={c.num_detectors} "
              f"(expected {num_detectors(3, d)})")
        det = c.compile_detector_sampler().sample(shots=256,
                                                  append_observables=True)
        print(f"  noiseless detectors+observable all-zero: "
              f"{np.asarray(det).sum() == 0}")
