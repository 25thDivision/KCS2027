"""NVIDIA pre-decoder wrapper (decoders/nvidia_predecoder.py), surface.

Checked against the NVIDIA clone's own code (./Ising-decoding/code):
  * lattice: which NVIDIA orientation equals this repository's code
  * input: bit-identical (B, 4, T, D, D) tensors, both from this
    repository's Stim samples (NVIDIA detector transform) and from
    NVIDIA's Stim samples (NVIDIA inference datapipe)
  * residual: identical to NVIDIA's evaluation module for random logits
Checkpoint-dependent tests skip when no checkpoint file is present.
"""
import types

import numpy as np
import pytest
import torch

from baseline.mwpm import build_matching
from circuits.rotatedSurface.rotatedSurface import surface_def
from dataset_generation.rotatedSurface_stim import (
    build_rotatedSurface_stim_circuit, split_rotatedSurface_sample,
    detectors_from_dataset_rotatedSurface, logical_label_rotatedSurface)
from decoders import nvidia_predecoder as nv

pytestmark = pytest.mark.skipif(
    not nv.NVIDIA_CODE.exists(), reason="Ising-decoding clone not present")

D, T = 3, 3
NOISE = "realistic/dp0.005_mf0.02_rf0.02_gd0.015"
ROTATIONS = ["XV", "XH", "ZV", "ZH"]            # O1, O2, O3, O4
CHECKPOINTS = sorted((nv.NVIDIA_CODE.parent / "models").glob("*.safetensors"))


def our_samples(shots=512, seed=7, d=D):
    """Noisy samples of this repository's abstract surface circuit ->
    (check values, final data bits)."""
    c = build_rotatedSurface_stim_circuit(T, "X", 0.01, NOISE, distance=d)
    raw = c.compile_sampler(seed=seed).sample(shots=shots)
    syn, dat = split_rotatedSurface_sample(raw, T, d)
    return syn, dat


def supports(lay, ctype):
    return [frozenset(np.nonzero(row)[0].tolist()) for row in lay.H[ctype]]


def nvidia_supports(rotation, d=D):
    """(X supports, Z supports) of NVIDIA's SurfaceCode in stabilizer
    order. Call inside nv.nvidia_imports()."""
    from qec.surface_code.memory_circuit import SurfaceCode
    code = SurfaceCode(d, first_bulk_syndrome_type=rotation[0],
                       rotated_type=rotation[1])
    return ([frozenset(np.nonzero(r)[0].tolist()) for r in code.hx],
            [frozenset(np.nonzero(r)[0].tolist()) for r in code.hz])


def to_nvidia_order(lay, rotation):
    """Columns of this repository's detector vector in NVIDIA's detector
    order: Z cycle 0, then [X, Z] per cycle, then the boundary Z."""
    hx, hz = nvidia_supports(rotation, lay.distance)
    ours = {t: supports(lay, t) for t in ("Z", "X")}
    z_names = [lay.names["Z"][ours["Z"].index(s)] for s in hz]
    x_names = [lay.names["X"][ours["X"].index(s)] for s in hx]
    index = lay.detector_index(T)
    cols = [index[("Z", n, 0)] for n in z_names]
    for cyc in range(1, T):
        cols += [index[("X", n, cyc)] for n in x_names]
        cols += [index[("Z", n, cyc)] for n in z_names]
    cols += [index[("F", n)] for n in z_names]
    return cols, x_names, z_names


def test_detector_index_matches_generator():
    """detector_index reproduces the generator's detector order."""
    lay = nv.surface_layout(D)
    syn, dat = our_samples(64)
    det = detectors_from_dataset_rotatedSurface(syn, dat, T, D)
    cm = syn.reshape(-1, T, lay.n_checks)
    for key, k in lay.detector_index(T).items():
        if key[0] == "F":
            name = key[1]
            row = lay.H["Z"][lay.names["Z"].index(name)]
            want = (dat @ row) % 2 ^ cm[:, T - 1, lay.col[name]]
        else:
            _, name, cyc = key
            want = cm[:, cyc, lay.col[name]].copy()
            if cyc >= 1:
                want ^= cm[:, cyc - 1, lay.col[name]]
        assert np.array_equal(det[:, k], want), key


@pytest.mark.parametrize("d", [3, 5])
def test_orientation_is_o4(d):
    """Supports and logical Z equal NVIDIA's ZH (O4) with no relabeling;
    the other three orientations do not match."""
    lay = nv.surface_layout(d)
    ours = (set(supports(lay, "X")), set(supports(lay, "Z")))
    matches = []
    with nv.nvidia_imports():
        for rot in ROTATIONS:
            hx, hz = nvidia_supports(rot, d)
            if (set(hx), set(hz)) == ours:
                matches.append(rot)
    assert matches == ["ZH"]
    assert nv.ORIENTATION == "ZH"
    assert lay.logical_z == list(range(0, d * d, d))     # column 0


@pytest.mark.parametrize("D", [3, 5])
def test_input_matches_nvidia_transform(D):
    """This repository's samples through nvidia_input == through NVIDIA's
    detector transforms (both implementations in the clone)."""
    lay = nv.surface_layout(D)
    syn, dat = our_samples(d=D)
    mine = nv.nvidia_input(syn, T, D)
    det = detectors_from_dataset_rotatedSurface(syn, dat, T, D)
    with nv.nvidia_imports():
        from data.predecoder_transform import dets_to_predecoder_inputs
        from qec.surface_code.detector_input import (
            SurfaceDetectorInputTransform)
        cols, _, _ = to_nvidia_order(lay, nv.ORIENTATION)
        dets = torch.from_numpy(det[:, cols])
        ref, _, _ = dets_to_predecoder_inputs(
            dets, distance=D, n_rounds=T, basis="Z",
            code_rotation=nv.ORIENTATION)
        ref2 = SurfaceDetectorInputTransform(
            distance=D, rounds=T, basis="Z", rotation=nv.ORIENTATION)(dets)
    assert mine.shape == (syn.shape[0], 4, T, D, D)
    assert mine.dtype == np.float32
    assert mine[:, :2].any()                 # the sample is not trivial
    assert np.array_equal(mine, ref.numpy())
    assert np.array_equal(mine, ref2.numpy())


@pytest.mark.parametrize("D", [3, 5])
def test_input_matches_nvidia_datapipe(D):
    """NVIDIA's own noisy Stim samples (its inference datapipe): the
    measurement record converted to this repository's check values gives
    the datapipe's trainX bit for bit."""
    lay = nv.surface_layout(D)
    shots = 256
    with nv.nvidia_imports():
        from data.datapipe_stim import QCDataPipePreDecoder_Memory_inference
        torch.manual_seed(0)
        pipe = QCDataPipePreDecoder_Memory_inference(
            distance=D, n_rounds=T, num_samples=shots,
            error_mode="circuit_level_surface_custom", p_error=0.01,
            measure_basis="Z", code_rotation=nv.ORIENTATION)
        ref = torch.stack([pipe[i]["trainX"] for i in range(shots)]).numpy()
        _, x_names, z_names = to_nvidia_order(lay, nv.ORIENTATION)
    # NVIDIA measures [X checks, Z checks] per round with reset, so a
    # measurement is the check value
    meas = pipe.meas.numpy()                      # (shots, T, D*D - 1)
    cm = np.zeros((shots, T, lay.n_checks), dtype=np.uint8)
    for i, name in enumerate(x_names + z_names):
        cm[:, :, lay.col[name]] = meas[:, :, i]
    mine = nv.nvidia_input(cm.reshape(shots, -1), T, D)
    assert ref[:, :2].any()
    assert np.array_equal(mine, ref)


class _FixedLogits(torch.nn.Module):
    """Stand-in model: returns preset logits whatever the input."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", logits)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.logits[:x.shape[0]]


@pytest.mark.parametrize("D", [3, 5])
def test_residual_matches_nvidia_eval_module(D):
    """Random logits: detector_correction gives the residual and logical
    frame of NVIDIA's PreDecoderMemoryEvalModule. NVIDIA drops the X
    detectors of the last cycle from its residual (they are masked at the
    input); this repository keeps them, so that block is compared after
    removing the baseline bit."""
    lay = nv.surface_layout(D)
    syn, dat = our_samples(256, d=D)
    det = detectors_from_dataset_rotatedSurface(syn, dat, T, D)
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(syn.shape[0], 4, T, D, D, generator=g)
    corr, pre_l = nv.detector_correction(logits.numpy(), T, D)
    mine = det ^ corr
    cfg = types.SimpleNamespace(
        distance=D, enable_fp16=False,
        test=types.SimpleNamespace(meas_basis_test="Z", th_data=0.0,
                                   th_syn=0.0, n_rounds=T),
        data=types.SimpleNamespace(code_rotation=nv.ORIENTATION))
    with nv.nvidia_imports():
        from evaluation.logical_error_rate import (
            PreDecoderMemoryEvalModule, _build_stab_maps)
        cols, x_names, _ = to_nvidia_order(lay, nv.ORIENTATION)
        module = PreDecoderMemoryEvalModule(
            _FixedLogits(logits), cfg, _build_stab_maps(D, nv.ORIENTATION),
            torch.device("cpu"))
        out = module(torch.from_numpy(det[:, cols])).numpy()
    assert np.array_equal(pre_l, out[:, 0])
    index = lay.detector_index(T)
    last_x = [index[("X", n, T - 1)] for n in x_names]
    expect = mine.copy()
    expect[:, last_x] ^= det[:, last_x]
    assert np.array_equal(expect[:, cols], out[:, 1:])


def test_negative_logits_reduce_to_mwpm():
    """No correction predicted -> residual = detectors, logical frame 0."""
    syn, dat = our_samples(64)
    logits = -np.ones((syn.shape[0], 4, T, D, D), dtype=np.float32)
    corr, pre_l = nv.detector_correction(logits, T, D)
    assert not corr.any() and not pre_l.any()


@pytest.mark.parametrize("cycle", range(T))
@pytest.mark.parametrize("qubit", range(D * D))
def test_single_x_correction(qubit, cycle):
    """One X correction on one data qubit at one cycle flips exactly the
    Z detectors of that cycle whose support holds the qubit, and the
    logical frame iff the qubit is on logical Z."""
    lay = nv.surface_layout(D)
    logits = -np.ones((1, 4, T, D, D), dtype=np.float32)
    logits[0, 1, cycle, qubit // D, qubit % D] = 1.0
    corr, pre_l = nv.detector_correction(logits, T, D)
    index = lay.detector_index(T)
    want = np.zeros_like(corr)
    for i, name in enumerate(lay.names["Z"]):
        if lay.H["Z"][i, qubit]:
            want[0, index[("Z", name, cycle)]] = 1
    assert np.array_equal(corr, want)
    assert int(pre_l[0]) == int(qubit in surface_def(D).LOGICAL_Z_IDX)


@pytest.mark.parametrize("ctype", ["Z", "X"])
def test_single_measurement_correction(ctype):
    """A check-measurement correction at cycle t flips that check's
    detectors at cycles t and t+1 (the time-like pair)."""
    lay = nv.surface_layout(D)
    index = lay.detector_index(T)
    channel = 3 if ctype == "Z" else 2
    for name in lay.names[ctype]:
        r, c = lay.cell[name]
        for cyc in range(T):
            logits = -np.ones((1, 4, T, D, D), dtype=np.float32)
            logits[0, channel, cyc, r, c] = 1.0
            corr, pre_l = nv.detector_correction(logits, T, D)
            want = np.zeros_like(corr)
            for t in (cyc, cyc + 1):
                if (ctype, name, t) in index:
                    want[0, index[(ctype, name, t)]] = 1
            assert np.array_equal(corr, want), (name, cyc)
            assert not pre_l.any()


@pytest.mark.skipif(not CHECKPOINTS, reason="no NVIDIA checkpoint file")
@pytest.mark.parametrize("ckpt", CHECKPOINTS, ids=lambda p: p.name)
def test_checkpoint_noiseless_and_decode(ckpt):
    """Real weights: a noiseless input predicts no correction, and the
    full path returns one logical flip per shot. The LER itself is a
    recorded quantity, not an acceptance criterion."""
    model = nv.load_predecoder(ckpt, "cpu")
    assert next(model.parameters()).dtype == torch.float32
    c0 = build_rotatedSurface_stim_circuit(T, distance=D)
    raw = c0.compile_sampler(seed=1).sample(shots=64)
    syn0, _ = split_rotatedSurface_sample(raw, T, D)
    logits = nv.predecoder_logits(model, nv.nvidia_input(syn0, T, D))
    corr, pre_l = nv.detector_correction(logits, T, D)
    assert not corr.any() and not pre_l.any()

    syn, dat = our_samples(2000)
    matching = build_matching(T, "X", 0.01, NOISE, "surface", D)
    pred, det, residual, pre_l = nv.nvidia_mwpm_decode(
        model, syn, dat, T, matching, D, return_parts=True)
    y = logical_label_rotatedSurface(dat, D)
    assert pred.shape == y.shape and set(np.unique(pred)) <= {0, 1}
    assert residual.shape == det.shape
