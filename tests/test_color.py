"""Triangular color code: definitions (circuits/color), Stim generator
(dataset_generation/color_stim.py), detector graph and the restriction
decoder baseline (baseline/mwpm_color.py)."""
import numpy as np
import pytest

from baseline.mwpm import build_matching, mwpm_ler_from_hardware
from baseline.mwpm_color import RestrictionDecoder
from circuits.color.colorcode import (
    ANCILLA_MODES, ColorCodeHardware, check_values, color_def)
from dataset_generation import color_stim as cs
from model.graph import GraphBuilder, augment_features

NOISE = "realistic/dp0.005_mf0.02_rf0.02_gd0.015"


def test_labeling_is_the_kcs_hardware_definition():
    """Confirmed decision 8: faces and logicals of KCS STEANE_CODE /
    D5_COLOR_CODE."""
    S3, S5 = color_def(3), color_def(5)
    assert list(zip(S3.FACE_COLORS, S3.FACES)) == [
        ("R", (0, 1, 2, 3)), ("G", (1, 2, 4, 5)), ("B", (2, 3, 5, 6))]
    assert S3.LOGICAL_Z_IDX == [0, 1, 4] and S3.LOGICAL_X_IDX == [0, 2, 5]
    assert S3.NUM_DATA == 7
    assert list(zip(S5.FACE_COLORS, S5.FACES)) == [
        ("R", (1, 2, 5, 6)), ("R", (3, 4, 7, 8)), ("G", (0, 1, 5, 9)),
        ("G", (2, 3, 6, 7, 10, 11)), ("B", (5, 6, 9, 10, 12, 13)),
        ("B", (7, 8, 11, 14)), ("R", (10, 11, 13, 14, 15, 16)),
        ("G", (12, 13, 15, 17)), ("B", (15, 16, 17, 18))]
    assert S5.LOGICAL_Z_IDX == S5.LOGICAL_X_IDX == [0, 1, 2, 3, 4]
    assert S5.NUM_DATA == 19


@pytest.mark.parametrize("d", [3, 5])
def test_definition_consistency(d):
    S = color_def(d)
    h = S.parity_check()
    # face supports commute with each other (self-dual CSS)
    assert not ((h @ h.T) % 2).any()
    lz = np.zeros(S.NUM_DATA, dtype=int)
    lz[S.LOGICAL_Z_IDX] = 1
    lx = np.zeros(S.NUM_DATA, dtype=int)
    lx[S.LOGICAL_X_IDX] = 1
    # logical Z commutes with every X face, logical X with every Z face,
    # and the two logicals anticommute
    assert not ((h @ lz) % 2).any() and not ((h @ lx) % 2).any()
    assert (lz @ lx) % 2 == 1
    # one logical qubit: n - 2 * rank(H) = 1
    rank = np.linalg.matrix_rank(h)            # full rank over the reals
    assert rank == S.N_FACES and S.NUM_DATA - 2 * S.N_FACES == 1
    # three-colorable: faces of one color are disjoint
    for color in "RGB":
        rows = [i for i, c in enumerate(S.FACE_COLORS) if c == color]
        assert h[rows].sum(axis=0).max() <= 1
    # geometry: faces are compact in the data coordinates and the grid
    # cells are distinct
    for face in S.FACES:
        xs = [S.DATA_COORDS[q][0] for q in face]
        ys = [S.DATA_COORDS[q][1] for q in face]
        assert max(xs) - min(xs) <= 2.0 and max(ys) - min(ys) <= 2.0
    cells = [S.ANC_GRID[n] for n in S.Z_NAMES]
    assert len(set(cells)) == S.N_FACES
    assert all(S.ANC_GRID[f"Z{i}"] == S.ANC_GRID[f"X{i}"]
               for i in range(S.N_FACES))
    assert S.CYCLE_ORDER == S.Z_NAMES + S.X_NAMES


def test_grid_cells():
    assert color_def(3).GRID_SHAPE == (2, 2)
    assert [color_def(3).ANC_GRID[f"Z{i}"] for i in range(3)] == [
        (0, 0), (0, 1), (1, 1)]
    assert color_def(5).GRID_SHAPE == (5, 4)
    assert [color_def(5).ANC_GRID[f"Z{i}"] for i in range(9)] == [
        (0, 1), (0, 3), (0, 0), (1, 2), (2, 1), (1, 3), (3, 2), (3, 1),
        (4, 2)]


@pytest.mark.parametrize("d", [3, 5])
def test_generator_shapes(d):
    S = color_def(d)
    c0 = cs.build_color_stim_circuit(3, distance=d)
    assert c0.num_detectors == cs.num_detectors(3, d)
    assert not c0.compile_detector_sampler(seed=0).sample(
        128, append_observables=True).any()
    c = cs.build_color_stim_circuit(3, "X", 0.01, NOISE, distance=d)
    syn, dat = cs.sample_flips_color(c, 400, 3, seed=1, distance=d)
    assert syn.shape == (400, 3 * S.N_CHECKS)
    assert dat.shape == (400, S.NUM_DATA)
    t = cs.syndrome_tensor_color(syn, 3, d)
    assert t.shape == (400, 6, *S.GRID_SHAPE)
    a = cs.detectors_from_dataset_color(syn, dat, 3, d)
    b = cs.detectors_from_tensor_color(t, dat, d)
    assert np.array_equal(a, b) and a.any()
    assert a.shape[1] == cs.num_detectors(3, d)
    # the flip-based detectors are the circuit's own detectors: same
    # seed, same shots
    y = cs.logical_label_color(dat, d)
    assert y.shape == (400,) and set(np.unique(y)) <= {0, 1}


@pytest.mark.parametrize("d, nodes", [(3, 18), (5, 54)])
def test_graph_builder(d, nodes):
    """d = 3, 3 cycles: Z 3x3 + X 3x2 + final-Z 3 = 18 nodes."""
    S = color_def(d)
    gb = GraphBuilder("color", 3, d)
    assert gb.num_nodes == nodes == cs.num_detectors(3, d)
    kinds = [k for k, _, _ in gb.nodes]
    assert kinds.count("Z") == 3 * S.N_FACES
    assert kinds.count("X") == 2 * S.N_FACES
    assert kinds.count("final") == S.N_FACES
    assert np.array_equal(gb.adjacency, gb.adjacency.T)
    assert gb.adjacency.sum(axis=1).min() >= 1
    # node values of the augmented tensor = the generator's detectors
    import torch
    c = cs.build_color_stim_circuit(3, "X", 0.01, NOISE, distance=d)
    syn, dat = cs.sample_flips_color(c, 200, 3, seed=2, distance=d)
    t = cs.syndrome_tensor_color(syn, 3, d)
    x = augment_features(torch.from_numpy(t), torch.from_numpy(dat),
                         "color", d)
    det = cs.detectors_from_dataset_color(syn, dat, 3, d)
    assert np.array_equal(gb.node_values(x).numpy(), det)


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("mode", ANCILLA_MODES)
def test_hardware_circuit(d, mode):
    S = color_def(d)
    qc = ColorCodeHardware(3, d, mode).build_circuit()
    assert qc.num_qubits == S.num_qubits(3, mode)
    ops = qc.count_ops()
    assert ops["measure"] == 3 * S.N_CHECKS + S.NUM_DATA
    assert ops["cx"] == 3 * 2 * sum(len(f) for f in S.FACES)
    if mode == "fresh":
        assert "reset" not in ops
        assert qc.num_qubits == S.NUM_DATA + 3 * S.N_CHECKS
    else:
        assert ops["reset"] == 3 * S.N_CHECKS - 2
        assert qc.num_qubits == S.NUM_DATA + 2
    assert ColorCodeHardware(3, 3).ancilla == "fresh"
    assert ColorCodeHardware(3, 5).ancilla == "reuse"


def test_check_values_layout():
    S = color_def(3)
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 2, (20, 3 * S.N_CHECKS))
    vals = check_values(raw, 3, 3)
    mat = cs.check_matrix_from_dict_color(vals, 3, 3)
    assert np.array_equal(mat, raw)


@pytest.mark.parametrize("d", [3, 5])
def test_restriction_decoder(d):
    S = color_def(d)
    dec = RestrictionDecoder(d)
    assert dec.table.shape == (2 ** S.N_FACES, S.NUM_DATA)
    # every single data error is corrected exactly
    single = np.eye(S.NUM_DATA, dtype=np.uint8)
    assert np.array_equal(dec.decode_batch(dec.syndrome(single)), single)
    # no error -> no correction
    assert not dec.table[0].any()
    # corrections that are returned for a consistent syndrome reproduce it
    k = np.arange(2 ** S.N_FACES)
    syn = ((k[:, None] >> np.arange(S.N_FACES)) & 1).astype(np.uint8)
    got = dec.syndrome(dec.decode_batch(syn))
    assert np.array_equal(got[dec.consistent], syn[dec.consistent])
    if d == 3:
        assert dec.consistent.all()
    else:
        assert 0 < dec.consistent.sum() < len(k)     # heuristic at d = 5


@pytest.mark.parametrize("d", [3, 5])
def test_mwpm_baseline_beats_raw(d):
    c = cs.build_color_stim_circuit(
        3, "X", 0.005, "realistic/dp0.001_mf0.01_rf0.01_gd0.008",
        distance=d)
    syn, dat = cs.sample_flips_color(c, 20000, 3, seed=3, distance=d)
    y = cs.logical_label_color(dat, d)
    decoder = build_matching(3, "X", 0.005, NOISE, "color", d)
    assert isinstance(decoder, RestrictionDecoder)
    l, pred = mwpm_ler_from_hardware(syn, dat, 3, decoder, "color",
                                     return_pred=True, distance=d)
    assert pred.shape == y.shape
    assert l < float(y.mean())
