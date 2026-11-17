"""Hybrid decoders (decoders/hybrid.py): MWPM+ML and ML+MWPM."""
import numpy as np
import pytest
import torch

from decoders.hybrid import (
    FinalSyndromeMatcher, hybrid_predictions, qubit_correction)
from model import code_spec, get_model_class, get_model_module

CYCLES = 3
CASES = [("heavyhex", 3), ("surface", 3), ("surface", 5), ("color", 3),
         ("color", 5)]
PARAMS = [pytest.param(c, d, id=f"{c}{d}") for c, d in CASES]


def exact(code, distance):
    """True when the per-qubit decoder reproduces EVERY syndrome (the
    color d = 5 restriction decoder only does so for part of them)."""
    return not (code == "color" and distance == 5)


class _FixedQubits(torch.nn.Module):
    """Stand-in model: per-qubit logits from fn(x) -> (B, num_qubits)
    0/1 mask (+1 where the mask is set, -1 elsewhere)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mask = torch.as_tensor(self.fn(x)).float()
        return mask * 2 - 1, torch.zeros(x.shape[0], 1)


def zeros_like_features(code, distance, shots):
    h, w = code_spec(code, distance)["grid"]
    return np.zeros((shots, 2 * CYCLES, h, w), dtype=np.uint8)


def random_bits(code, distance, shots, seed=0):
    nq = code_spec(code, distance)["num_qubits"]
    return np.random.default_rng(seed).integers(
        0, 2, (shots, nq), dtype=np.uint8)


@pytest.mark.parametrize("code, distance", PARAMS)
def test_matcher_definition(code, distance):
    m = FinalSyndromeMatcher(code, distance)
    nq = code_spec(code, distance)["num_qubits"]
    assert m.H.shape[1] == nq and m.H.any(axis=1).all()
    if code == "color":            # up to three faces per qubit
        assert set(np.unique(m.H.sum(axis=0))) <= {1, 2, 3}
    else:                          # matchable: one or two Z checks
        assert set(np.unique(m.H.sum(axis=0))) <= {1, 2}
    assert m.logical_z and max(m.logical_z) < nq


@pytest.mark.parametrize("code, distance", PARAMS)
def test_residual_slots_match_prepare_features(code, distance):
    """residual_input puts H_z.b where prepare_features puts the
    final-Z detector bits of an all-zero syndrome history."""
    m = FinalSyndromeMatcher(code, distance)
    mod = get_model_module("cnn")
    bits = random_bits(code, distance, 64)
    want = mod.prepare_features(zeros_like_features(code, distance, 64),
                                bits, code, distance)
    got = m.residual_input(m.syndrome(bits), CYCLES)
    assert np.array_equal(np.asarray(want), got)


@pytest.mark.parametrize("code, distance", PARAMS)
def test_decode_reproduces_syndrome(code, distance):
    """Every syndrome is matched exactly (color d = 5: the syndromes of
    single errors)."""
    m = FinalSyndromeMatcher(code, distance)
    bits = random_bits(code, distance, 2000, seed=1)
    if not exact(code, distance):
        bits = np.eye(m.num_qubits, dtype=np.uint8)
    s = m.syndrome(bits)
    c = m.decode(s)
    assert c.shape == bits.shape
    assert np.array_equal(m.syndrome(c), s)


@pytest.mark.parametrize("code, distance", PARAMS)
def test_single_data_error(code, distance):
    """A single final data error: the MWPM stage returns a correction
    with the same syndrome and the same logical parity."""
    m = FinalSyndromeMatcher(code, distance)
    nq = code_spec(code, distance)["num_qubits"]
    err = np.eye(nq, dtype=np.uint8)
    s = m.syndrome(err)
    assert s.any(axis=1).all()                 # every error is detected
    c = m.decode(s)
    assert np.array_equal(m.syndrome(c), s)
    assert np.array_equal(m.logical(c), m.logical(err))
    assert (c.sum(axis=1) == 1).all()


@pytest.mark.parametrize("code, distance", PARAMS)
def test_no_error_gives_no_flip(code, distance):
    """Error-free input with a model that predicts nothing: LER 0."""
    nq = code_spec(code, distance)["num_qubits"]
    model = _FixedQubits(lambda x: torch.zeros(x.shape[0], nq))
    feats = zeros_like_features(code, distance, 32)
    bits = np.zeros((32, nq), dtype=np.uint8)
    out = hybrid_predictions(model, get_model_module("cnn"), feats, bits,
                             code, distance)
    for key in ("mwpm_ml", "ml_mwpm", "mwpm"):
        assert out[key].shape == (32,) and not out[key].any(), key


@pytest.mark.parametrize("code, distance", PARAMS)
def test_ml_then_mwpm_completes_the_model(code, distance):
    """ML+MWPM: whatever the model predicts, the combined correction
    reproduces the final syndrome; a model that predicts the true flips
    gives the true logical flip."""
    m = FinalSyndromeMatcher(code, distance)
    nq = code_spec(code, distance)["num_qubits"]
    mod = get_model_module("cnn")
    feats = zeros_like_features(code, distance, 500)
    bits = random_bits(code, distance, 500, seed=2)
    s = m.syndrome(bits)

    if exact(code, distance):
        guess = random_bits(code, distance, 500, seed=3)
        c2 = m.decode(s ^ m.syndrome(guess))
        assert np.array_equal(m.syndrome(guess ^ c2), s)

    truth = torch.from_numpy(bits)
    oracle = _FixedQubits(lambda x: truth[:x.shape[0]])
    out = hybrid_predictions(oracle, mod, feats, bits, code, distance,
                             batch_size=10 ** 6)
    assert np.array_equal(out["ml_mwpm"], m.logical(bits))
    # the same through precomputed logits
    out2 = hybrid_predictions(oracle, mod, feats, bits, code, distance,
                              q_logits=bits.astype(np.float32) * 2 - 1)
    assert np.array_equal(out2["ml_mwpm"], out["ml_mwpm"])


def random_features(code, distance, shots, seed):
    h, w = code_spec(code, distance)["grid"]
    return np.random.default_rng(seed).integers(
        0, 2, (shots, 2 * CYCLES, h, w), dtype=np.uint8)


def features_with_last_z(code, distance, z_last):
    """All-zero syndrome tensor except the last cycle's Z-check values."""
    m = FinalSyndromeMatcher(code, distance)
    feats = zeros_like_features(code, distance, z_last.shape[0])
    for i, (r, c) in enumerate(m.cells):
        feats[:, 2 * (CYCLES - 1), r, c] = z_last[:, i]
    return feats


@pytest.mark.parametrize("code, distance", PARAMS)
def test_mwpm_ml_residual_is_the_final_z_detector(code, distance):
    """Stage 1 decodes the last-cycle Z-check values; the residual handed
    to the model is exactly the final-Z detector channel that
    prepare_features computes on the real input."""
    m = FinalSyndromeMatcher(code, distance)
    mod = get_model_module("cnn")
    feats = random_features(code, distance, 300, seed=6)
    bits = random_bits(code, distance, 300, seed=7)
    z_last = m.last_cycle_syndrome(feats)
    c1 = m.decode(z_last)
    r = m.syndrome(bits) ^ m.syndrome(c1)
    assert r.any()                              # not a vacuous check
    if not exact(code, distance):
        return        # c1 need not reproduce z_last; r is still H.(f ^ c1)
    assert np.array_equal(r, m.syndrome(bits) ^ z_last)
    real = np.asarray(mod.prepare_features(feats, bits, code, distance))
    assert np.array_equal(m.residual_input(r, CYCLES)[:, -1], real[:, -1])


@pytest.mark.parametrize("code, distance", PARAMS)
def test_mwpm_ml_skips_the_model_without_residual(code, distance):
    """Errors that the last cycle already saw (z_last = H_z . final):
    the matching alone decides and the model is not called."""
    m = FinalSyndromeMatcher(code, distance)
    nq = code_spec(code, distance)["num_qubits"]
    err = np.eye(nq, dtype=np.uint8)            # every single data error
    feats = features_with_last_z(code, distance, m.syndrome(err))

    def never(x):
        raise AssertionError("the model must not run when r = 0")

    out = hybrid_predictions(_FixedQubits(never), get_model_module("cnn"),
                             feats, err, code, distance,
                             q_logits=np.zeros((nq, nq), dtype=np.float32))
    assert np.array_equal(out["mwpm_ml"], m.logical(err))
    assert np.array_equal(out["mwpm"], m.logical(err))


@pytest.mark.parametrize("code, distance", PARAMS)
def test_mwpm_ml_depends_on_the_model(code, distance):
    """Errors after the last cycle (z_last = 0): the whole syndrome is
    residual, so the result is the model's. A model that solves the
    residual gives the true logical flip; one that predicts nothing does
    not, on the shots where the error flips the logical."""
    m = FinalSyndromeMatcher(code, distance)
    nq = code_spec(code, distance)["num_qubits"]
    mod = get_model_module("cnn")
    err = np.eye(nq, dtype=np.uint8)
    feats = zeros_like_features(code, distance, nq)
    q_logits = np.zeros((nq, nq), dtype=np.float32)

    def solve(x):                                # reads the final-Z slots
        x = np.asarray(x.cpu())
        r = np.stack([x[:, -1, rr, cc] for rr, cc in m.cells], axis=1)
        return m.decode(r)

    good = hybrid_predictions(_FixedQubits(solve), mod, feats, err, code,
                              distance, q_logits=q_logits)
    idle = hybrid_predictions(
        _FixedQubits(lambda x: torch.zeros(x.shape[0], nq)), mod, feats,
        err, code, distance, q_logits=q_logits)
    assert np.array_equal(good["mwpm_ml"], m.logical(err))
    assert not idle["mwpm_ml"].any()
    assert not np.array_equal(good["mwpm_ml"], idle["mwpm_ml"])
    assert not good["mwpm"].any()                # stage 1 saw nothing


@pytest.mark.parametrize("name", ["cnn", "gnn"])
@pytest.mark.parametrize("code, distance", PARAMS)
def test_shapes_with_registry_models(name, code, distance):
    """Registry models through both hybrids: one 0/1 flip per shot."""
    spec = code_spec(code, distance)
    torch.manual_seed(0)
    model = get_model_class(name)(in_channels=2 * CYCLES,
                                  num_qubits=spec["num_qubits"],
                                  code=code, distance=distance)
    mod = get_model_module(name)
    h, w = spec["grid"]
    rng = np.random.default_rng(4)
    feats = rng.integers(0, 2, (40, 2 * CYCLES, h, w), dtype=np.uint8)
    bits = random_bits(code, distance, 40, seed=5)
    x = mod.prepare_features(feats, bits, code, distance)
    assert qubit_correction(model, x).shape == (40, spec["num_qubits"])
    out = hybrid_predictions(model, mod, feats, bits, code, distance)
    for key in ("mwpm_ml", "ml_mwpm", "mwpm"):
        assert out[key].shape == (40,)
        assert out[key].dtype == np.uint8
        assert set(np.unique(out[key])) <= {0, 1}
