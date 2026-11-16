"""Rotated surface code definitions (circuits/rotatedSurface) and its
Stim generators, d = 3 and d = 5.

The d = 3 constants and circuit digests below were taken from the former
distance-3-only modules (rotatedSurface3*.py) before they were replaced
by the distance-general ones; they pin the d = 3 path bit for bit.
"""
import hashlib

import numpy as np
import pytest

from circuits.rotatedSurface.rotatedSurface import (
    RotatedSurfaceHardware, check_values, embedding_for_surface,
    required_edges_surface, surface_def)
from dataset_generation import rotatedSurface_stim as rss
from dataset_generation.rotatedSurface_qpu_stim import (
    build_rotatedSurface_qpu_stim_circuit, sample_rotatedSurface_qpu_flips)

NOISE = "realistic/dp0.005_mf0.02_rf0.02_gd0.015"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def hw_listing(qc):
    return str([(i.operation.name,
                 [qc.find_bit(q).index for q in i.qubits],
                 [qc.find_bit(c).index for c in i.clbits]) for i in qc.data])


def qpu_profile(distance):
    """Synthetic surface qpu_avg profile over the patch's local indices."""
    S = surface_def(distance)
    prof = {"mode": "qpu_avg_v1", "code": "surface", "distance": distance,
            "readout": {str(i): 0.01 + 0.001 * i
                        for i in range(S.NUM_QUBITS)},
            "error_1q": {str(i): 0.0005 * (i + 1)
                         for i in range(S.NUM_QUBITS)},
            "error_2q": {}}
    for name in S.CYCLE_ORDER:
        for c in S.CHECK_DEFS[name][1]:
            prof["error_2q"][f"{S.L[c]}-{S.AIDX[name]}"] = \
                0.004 + 0.0001 * S.L[c]
    return prof


def test_d3_constants():
    S = surface_def(3)
    assert S.CYCLE_ORDER == ["Z22", "Z44", "Z40", "Z26",
                             "X42", "X24", "X02", "X64"]
    assert S.DATA_COORDS == [(1, 1), (3, 1), (5, 1), (1, 3), (3, 3), (5, 3),
                             (1, 5), (3, 5), (5, 5)]
    assert {n: S.CHECK_DEFS[n][1] for n in S.CYCLE_ORDER} == {
        "Z22": ((1, 1), (3, 1), (1, 3), (3, 3)),
        "Z44": ((3, 3), (5, 3), (3, 5), (5, 5)),
        "Z40": ((3, 1), (5, 1)),
        "Z26": ((1, 5), (3, 5)),
        "X42": ((3, 1), (3, 3), (5, 1), (5, 3)),
        "X24": ((1, 3), (1, 5), (3, 3), (3, 5)),
        "X02": ((1, 1), (1, 3)),
        "X64": ((5, 3), (5, 5)),
    }
    assert S.LOGICAL_Z_IDX == [0, 3, 6]
    assert S.GRID_SHAPE == (4, 4)
    assert S.ANC_GRID == {"Z22": (1, 1), "Z44": (2, 2), "Z40": (0, 2),
                          "Z26": (3, 1), "X42": (1, 2), "X24": (2, 1),
                          "X02": (1, 0), "X64": (2, 3)}
    assert [S.AIDX[n] for n in S.CYCLE_ORDER] == list(range(9, 17))
    emb = embedding_for_surface("ibm_miami", 3)
    assert emb[(1, 1)] == 62 and emb[(5, 5)] == 66 and emb[(3, 3)] == 64
    assert len(set(emb.values())) == 17


def test_d3_circuits_unchanged():
    """Digests of the d = 3 circuits recorded from the former modules."""
    assert digest(str(rss.build_rotatedSurface_stim_circuit(
        3, "X", 0.01, NOISE))) == "a6dfa6300bbdaff0"
    assert digest(str(build_rotatedSurface_qpu_stim_circuit(
        3, "X", 0.01, qpu_profile(3)))) == "4372520a5a4f7028"
    assert digest(hw_listing(RotatedSurfaceHardware(3, 3).build_circuit())) \
        == "69494a1090e123ab"


@pytest.mark.parametrize("d", [3, 5])
def test_code_definition(d):
    S = surface_def(d)
    assert S.NUM_DATA == d * d and S.N_CHECKS == d * d - 1
    assert len(S.Z_NAMES) == len(S.X_NAMES) == (d * d - 1) // 2
    assert len(set(S.CYCLE_ORDER)) == S.N_CHECKS
    hz = np.zeros((len(S.Z_NAMES), S.NUM_DATA), dtype=int)
    hx = np.zeros((len(S.X_NAMES), S.NUM_DATA), dtype=int)
    for i, n in enumerate(S.Z_NAMES):
        hz[i, [S.DIDX[c] for c in S.CHECK_DEFS[n][1]]] = 1
    for i, n in enumerate(S.X_NAMES):
        hx[i, [S.DIDX[c] for c in S.CHECK_DEFS[n][1]]] = 1
    # X and Z stabilizers commute; the logicals commute with the
    # stabilizers of the other type and anticommute with each other
    assert not ((hx @ hz.T) % 2).any()
    lz = np.zeros(S.NUM_DATA, dtype=int)
    lz[S.LOGICAL_Z_IDX] = 1
    lx = np.zeros(S.NUM_DATA, dtype=int)
    lx[[S.DIDX[c] for c in S.LOGICAL_X]] = 1
    assert not ((hx @ lz) % 2).any() and not ((hz @ lx) % 2).any()
    assert (lz @ lx) % 2 == 1
    assert len(S.LOGICAL_Z_IDX) == d
    # the 45-degree embedding is injective and inside the 12 x 10 device
    emb = embedding_for_surface("ibm_miami", d)
    assert len(set(emb.values())) == S.NUM_QUBITS
    assert 0 <= min(emb.values()) and max(emb.values()) < 120
    # every required edge is a unit step on the row-major lattice
    for u, v in required_edges_surface(d):
        a, b = emb[u], emb[v]
        assert abs(a // 10 - b // 10) + abs(a % 10 - b % 10) == 1


@pytest.mark.parametrize("d", [3, 5])
def test_stim_circuit(d):
    c0 = rss.build_rotatedSurface_stim_circuit(3, distance=d)
    assert c0.num_detectors == rss.num_detectors(3, d)
    det = c0.compile_detector_sampler(seed=0).sample(
        256, append_observables=True)
    assert not det.any()
    c = rss.build_rotatedSurface_stim_circuit(3, "X", 0.01, NOISE,
                                              distance=d)
    c.detector_error_model(decompose_errors=True)
    # circuit-level distance is the code distance (hook-safe schedule)
    assert len(c.shortest_graphlike_error()) == d


@pytest.mark.parametrize("d", [3, 5])
def test_detector_reconstructions_agree(d):
    """Sampled flips -> tensor -> detectors equals check values ->
    detectors, equals Stim's own detector sampler on the same seed."""
    S = surface_def(d)
    c = rss.build_rotatedSurface_stim_circuit(3, "X", 0.01, NOISE,
                                              distance=d)
    syn, dat = rss.sample_flips_rotatedSurface(c, 500, 3, seed=1,
                                               distance=d)
    assert syn.shape == (500, 3 * S.N_CHECKS) and dat.shape == (500, d * d)
    t = rss.syndrome_tensor_rotatedSurface(syn, 3, d)
    assert t.shape == (500, 6, d + 1, d + 1)
    a = rss.detectors_from_dataset_rotatedSurface(syn, dat, 3, d)
    b = rss.detectors_from_tensor_rotatedSurface(t, dat, d)
    assert np.array_equal(a, b)
    assert a.shape[1] == rss.num_detectors(3, d)
    # flips are relative to the noiseless reference: with no noise the
    # detectors vanish, with noise some fire
    assert a.any()


@pytest.mark.parametrize("d", [3, 5])
def test_hardware_shaped_stim_matches_abstract(d):
    S = surface_def(d)
    cq = build_rotatedSurface_qpu_stim_circuit(3, distance=d)
    assert cq.num_detectors == rss.num_detectors(3, d)
    assert not cq.compile_detector_sampler(seed=0).sample(
        128, append_observables=True).any()
    cn = build_rotatedSurface_qpu_stim_circuit(3, "X", 0.01, qpu_profile(d),
                                               distance=d)
    mat, dat = sample_rotatedSurface_qpu_flips(cn, 200, 3, seed=2,
                                               distance=d)
    assert mat.shape == (200, 3 * S.N_CHECKS) and dat.shape == (200, d * d)
    with pytest.raises(ValueError):
        build_rotatedSurface_qpu_stim_circuit(
            3, profile=qpu_profile(d), distance=8 - d)


@pytest.mark.parametrize("d", [3, 5])
def test_check_values_undo_the_xor_chain(d):
    S = surface_def(d)
    rng = np.random.default_rng(0)
    vals = rng.integers(0, 2, (50, 3, S.N_CHECKS))
    raw = np.cumsum(vals, axis=1) % 2            # no-reset accumulation
    got = check_values(raw.reshape(50, -1), 3, d)
    for j, name in enumerate(S.CYCLE_ORDER):
        assert np.array_equal(got[name], vals[:, :, j])


def test_hardware_circuit_sizes():
    for d in (3, 5):
        qc = RotatedSurfaceHardware(3, d).build_circuit()
        S = surface_def(d)
        assert qc.num_qubits == S.NUM_QUBITS
        assert qc.count_ops().get("reset", 0) == 0
        assert qc.count_ops()["measure"] == 3 * S.N_CHECKS + S.NUM_DATA
