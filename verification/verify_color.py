#!/usr/bin/env python3
"""
Contract gate for the triangular color code (--code color, d = 3 and 5)
=======================================================================
Dataset generation / hardware submission for --code color must NOT
proceed until this script prints ALL PASS. --distance picks one distance;
without it both distances are checked.

  [A] With zero noise, every abstract-Stim detector is deterministic
  [B] Stim raw-stream invariants: Z-check values are 0, X-check
      cycle-to-cycle XORs are 0, final data face parity == last Z-check
      value, logical Z parity is 0
  [C] The hardware circuit (noiseless Aer) satisfies the same invariants
      through colorcode.check_values(), with the same bit layout
      (cyc*N_CHECKS + j, CYCLE_ORDER), in BOTH ancilla modes (fresh
      ancillas, and two reused ancillas with mid-circuit reset)
  [D] For an injected single DATA error on every data qubit (X and Z),
      the set of checks firing at cycle 1 is identical between Stim and
      Aer (both ancilla modes) and equals the faces holding the qubit
  [E] Check-value equivalence: for the same injected errors the detector
      stream reconstructed from raw values is shot-independent, bit-
      identical between Stim and Aer (both ancilla modes), and actually
      fires (no vacuous pass)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from qiskit_aer import AerSimulator  # noqa: E402

from circuits.color.colorcode import (  # noqa: E402
    ColorCodeHardware, ANCILLA_MODES, check_values, color_def)
from dataset_generation.color_stim import (  # noqa: E402
    build_color_stim_circuit, split_color_sample,
    check_matrix_from_dict_color, detectors_from_dataset_color,
    num_detectors)

STIM_SHOTS = 512
AER_SHOTS = 300
CYCLES = 3
DET_EPS = 0.002
IDEAL = "ideal/dp0_mf0_rf0_gd0"
DISTANCES = (3, 5)

# set by run_gate() for the distance under test
DIST = 3
S = color_def(DIST)


def stim_run(inject=None):
    c = build_color_stim_circuit(CYCLES, "X", 0.0, IDEAL, inject=inject,
                                 distance=DIST)
    raw = c.compile_sampler().sample(shots=STIM_SHOTS)
    syn, dat = split_color_sample(raw, CYCLES, DIST)
    return syn, dat, np.ones(syn.shape[0])


def aer_run(mode, inject=None):
    qc = ColorCodeHardware(CYCLES, DIST, mode).build_circuit(0, inject)
    counts = AerSimulator(method="stabilizer").run(
        qc, shots=AER_SHOTS).result().get_counts()
    syn, dat, w = [], [], []
    for bs, cnt in counts.items():
        d, s = bs.split()
        syn.append([int(b) for b in s[::-1]])
        dat.append([int(b) for b in d[::-1]])
        w.append(cnt)
    syn = np.array(syn, dtype=np.uint8)
    dat = np.array(dat, dtype=np.uint8)
    mat = check_matrix_from_dict_color(check_values(syn, CYCLES, DIST),
                                       CYCLES, DIST)
    return mat, dat, np.array(w)


def p1(bits, w):
    return float((bits * w).sum() / w.sum())


def invariants(tag, mat, dat, w):
    fails = []
    pos = {name: j for j, name in enumerate(S.CYCLE_ORDER)}
    n = S.N_CHECKS
    for name in S.Z_NAMES:
        for c in range(CYCLES):
            if (p := p1(mat[:, c * n + pos[name]], w)) > DET_EPS:
                fails.append(f"Z:{name}@c{c}={p:.3f}")
    for name in S.X_NAMES:
        for c in range(CYCLES - 1):
            x0 = mat[:, c * n + pos[name]]
            x1 = mat[:, (c + 1) * n + pos[name]]
            if (p := p1(x0 ^ x1, w)) > DET_EPS:
                fails.append(f"Xxor:{name}@c{c}={p:.3f}")
    for name, supp in S.Z_STABS.items():
        par = np.bitwise_xor.reduce(dat[:, list(supp)], axis=1)
        if (p := p1(par ^ mat[:, (CYCLES - 1) * n + pos[name]],
                    w)) > DET_EPS:
            fails.append(f"final:{name}={p:.3f}")
    zl = np.bitwise_xor.reduce(dat[:, S.LOGICAL_Z_IDX], axis=1)
    if (p := p1(zl, w)) > DET_EPS:
        fails.append(f"Z_L={p:.3f}")
    ok = not fails
    print(f"[{tag}] invariants: {'PASS' if ok else 'FAIL ' + str(fails[:8])}")
    return ok


def fired_at_cycle1(mat, w):
    fired = []
    n = S.N_CHECKS
    for j, name in enumerate(S.CYCLE_ORDER):
        if name in S.Z_STABS:
            p = p1(mat[:, 1 * n + j], w)
        else:
            p = p1(mat[:, 1 * n + j] ^ mat[:, 0 * n + j], w)
        if p > 1 - DET_EPS:
            fired.append(name)
        elif p > DET_EPS:
            fired.append(f"{name}?{p:.2f}")
    return sorted(fired)


def det_stream(mat, dat):
    """(shot-independent?, first detector row) from raw check values.
    (stim's own detector sampler would absorb an injected Pauli into its
    reference, so the stream is rebuilt from the values instead.)"""
    det = detectors_from_dataset_color(mat, dat, CYCLES, DIST)
    return bool((det == det[0]).all()), det[0]


def run_gate(distance):
    """All sections for one distance -> True when everything passed."""
    global DIST, S
    DIST, S = distance, color_def(distance)
    print(f"===== color code d={distance} ({S.NUM_DATA} data, "
          f"{S.N_FACES} faces) =====")
    ok = True

    # [A] Stim detector determinism
    c = build_color_stim_circuit(CYCLES, "X", 0.0, IDEAL, distance=DIST)
    det = np.asarray(c.compile_detector_sampler().sample(
        shots=STIM_SHOTS, append_observables=True), dtype=np.uint8)
    a_ok = (det.sum() == 0
            and det.shape[1] == num_detectors(CYCLES, DIST) + 1)
    print(f"[A] Stim noiseless detectors+observable all-zero: "
          f"{'PASS' if a_ok else 'FAIL'} (shape={det.shape}, sum={det.sum()})")
    ok &= a_ok

    # [B] Stim raw-stream invariants (MR values ARE check values)
    syn, dat, w = stim_run()
    ok &= invariants("B:Stim", syn, dat, w)

    # [C] hardware circuit on noiseless Aer, both ancilla modes
    for mode in ANCILLA_MODES:
        a_mat, a_dat, a_w = aer_run(mode)
        assert a_mat.shape[1] == syn.shape[1] == S.N_CHECKS * CYCLES
        ok &= invariants(f"C:Aer {mode}", a_mat, a_dat, a_w)

    # [D] + [E] single data errors on every qubit (inject after cycle 0)
    print("[D/E] single data-error signatures and detector streams "
          "(Stim vs Aer fresh / reuse):")
    bad = []
    for pauli in ("X", "Z"):
        stabs = S.Z_STABS if pauli == "X" else S.X_STABS
        for q in range(S.NUM_DATA):
            inject = (pauli, q, 0)
            exp = sorted(n for n, s in stabs.items() if q in s)
            s_syn, s_dat, s_w = stim_run(inject)
            f_stim = fired_at_cycle1(s_syn, s_w)
            s_det_ok, s_det = det_stream(s_syn, s_dat)
            good = (f_stim == exp) and s_det_ok and bool(s_det.any())
            for mode in ANCILLA_MODES:
                a_mat, a_dat, a_w = aer_run(mode, inject)
                a_det_ok, a_det = det_stream(a_mat, a_dat)
                good &= (fired_at_cycle1(a_mat, a_w) == exp and a_det_ok
                         and np.array_equal(a_det, s_det))
            if not good:
                bad.append((pauli, q, f_stim, exp))
    d_ok = not bad
    ok &= d_ok
    print(f"  {2 * S.NUM_DATA} injected errors: "
          f"{'PASS' if d_ok else 'FAIL ' + str(bad[:6])}")

    print(f"d={distance}: {'PASS' if ok else 'FAILURES PRESENT'}\n")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--distance", type=int, choices=DISTANCES, default=None,
                    help="one distance (default: all of 3, 5)")
    args = ap.parse_args()
    todo = [args.distance] if args.distance else list(DISTANCES)
    ok = all([run_gate(d) for d in todo])
    print(f"OVERALL: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
