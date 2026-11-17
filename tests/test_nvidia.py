"""NVIDIA pre-decoder wrapper (decoders/nvidia_predecoder.py), surface.

Checked against the NVIDIA clone's own code (./Ising-decoding/code):
  * lattice: which NVIDIA orientation equals this repository's code
  * time axis: T cycles + the final-Z detector layer of this repository
    have the fault structure of NVIDIA's circuit with T + 1 rounds (its
    last round is noiseless)
  * input: bit-identical (B, 4, T + 1, D, D) tensors, both from this
    repository's Stim samples (NVIDIA detector transform) and from
    NVIDIA's Stim samples (NVIDIA inference datapipe)
  * residual: identical to NVIDIA's evaluation module for random logits
Checkpoint-dependent tests skip when no checkpoint file is present.
"""
import types

import numpy as np
import pymatching
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
R = T + 1                                       # NVIDIA rounds
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


def nvidia_names(lay, rotation):
    """This repository's check names in NVIDIA's stabilizer order
    (matched by support). Call inside nv.nvidia_imports()."""
    hx, hz = nvidia_supports(rotation, lay.distance)
    ours = {t: supports(lay, t) for t in ("Z", "X")}
    return ([lay.names["X"][ours["X"].index(s)] for s in hx],
            [lay.names["Z"][ours["Z"].index(s)] for s in hz])


def to_nvidia_dets(lay, det, x_names, z_names):
    """This repository's detector matrix -> NVIDIA's detector layout for
    T + 1 rounds: Z round 0, then [X, Z] for rounds 1 .. T, then the
    boundary block. Rounds < T are the cycles, Z round T is the final-Z
    layer; X round T and the boundary block have no counterpart (zero)."""
    index = lay.detector_index(T)
    zero = np.zeros((det.shape[0], len(z_names)), dtype=det.dtype)
    blocks = [det[:, [index[("Z", n, 0)] for n in z_names]]]
    for r in range(1, T + 1):
        if r < T:
            blocks.append(det[:, [index[("X", n, r)] for n in x_names]])
            blocks.append(det[:, [index[("Z", n, r)] for n in z_names]])
        else:
            blocks.append(zero)
            blocks.append(det[:, [index[("F", n)] for n in z_names]])
    blocks.append(zero)
    return np.concatenate(blocks, axis=1)


def z_fault_graph(circuit, label_of):
    """{frozenset of labels} over the DEM errors of a circuit, restricted
    to the detectors in label_of (detector index -> label)."""
    dem = circuit.detector_error_model(decompose_errors=False,
                                       flatten_loops=True)
    out = set()
    for inst in dem:
        if inst.type != "error":
            continue
        dets = frozenset(label_of[t.val] for t in inst.targets_copy()
                         if t.is_relative_detector_id()
                         and t.val in label_of)
        if dets:
            out.add(dets)
    return out


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


@pytest.mark.parametrize("d", [3, 5])
def test_fault_structure_matches_nvidia_circuit(d):
    """Which Z detectors one fault can flip together: this repository's
    circuit with T cycles + the final-Z layer has exactly the fault
    structure of NVIDIA's ZH circuit with T + 1 rounds (labels = Z-check
    support on the grid and time layer). NVIDIA's last round is
    noiseless, so its boundary detectors take part in no fault."""
    S = surface_def(d)
    lay = nv.surface_layout(d)
    noise = {"data_depol": 0.001, "meas_flip": 0.001, "reset_flip": 0.001,
             "gate_depol": 0.001}
    ours_c = build_rotatedSurface_stim_circuit(T, "X", 0.0, noise,
                                               distance=d)
    index = lay.detector_index(T)

    def cells(name):
        return frozenset(((y - 1) // 2, (x - 1) // 2)
                         for x, y in S.CHECK_DEFS[name][1])

    label = {}
    for name in S.Z_NAMES:
        for cyc in range(T):
            label[index[("Z", name, cyc)]] = (cells(name), cyc)
        label[index[("F", name)]] = (cells(name), T)
    ours = z_fault_graph(ours_c, label)

    with nv.nvidia_imports():
        from qec.surface_code.memory_circuit import MemoryCircuit
        mc = MemoryCircuit(distance=d, idle_error=0.001, sqgate_error=0.001,
                           tqgate_error=0.001, spam_error=0.001,
                           n_rounds=R, basis="Z", code_rotation="ZH",
                           add_boundary_detectors=True)
        mc.set_error_rates()
        hz = [frozenset((int(q) // d, int(q) % d) for q in np.nonzero(r)[0])
              for r in mc.code.hz]
        circuit = mc.stim_circuit
    half = len(hz)
    label = {i: (s, 0) for i, s in enumerate(hz)}
    for r in range(1, R):
        base = half + (r - 1) * 2 * half + half
        label.update({base + i: (s, r) for i, s in enumerate(hz)})
    boundary = {half + (R - 1) * 2 * half + i: (s, "boundary")
                for i, s in enumerate(hz)}
    theirs = z_fault_graph(circuit, {**label, **boundary})
    assert not any(t == "boundary" for e in theirs for _, t in e)
    assert ours == theirs
    assert any(T in {t for _, t in e} for e in ours)     # final layer used


@pytest.mark.parametrize("D", [3, 5])
def test_input_matches_nvidia_transform(D):
    """This repository's samples through nvidia_input == through NVIDIA's
    detector transforms (both implementations in the clone)."""
    lay = nv.surface_layout(D)
    syn, dat = our_samples(d=D)
    mine = nv.nvidia_input(syn, dat, T, D)
    det = detectors_from_dataset_rotatedSurface(syn, dat, T, D)
    with nv.nvidia_imports():
        from data.predecoder_transform import dets_to_predecoder_inputs
        from qec.surface_code.detector_input import (
            SurfaceDetectorInputTransform)
        x_names, z_names = nvidia_names(lay, nv.ORIENTATION)
        dets = torch.from_numpy(to_nvidia_dets(lay, det, x_names, z_names))
        ref, _, _ = dets_to_predecoder_inputs(
            dets, distance=D, n_rounds=R, basis="Z",
            code_rotation=nv.ORIENTATION)
        ref2 = SurfaceDetectorInputTransform(
            distance=D, rounds=R, basis="Z", rotation=nv.ORIENTATION)(dets)
    assert mine.shape == (syn.shape[0], 4, R, D, D)
    assert mine.dtype == np.float32
    assert mine[:, 0].any() and mine[:, 1, T].any()   # X and final-Z used
    assert np.array_equal(mine, ref.numpy())
    assert np.array_equal(mine, ref2.numpy())


@pytest.mark.parametrize("D", [3, 5])
def test_input_matches_nvidia_datapipe(D):
    """NVIDIA's own noisy Stim samples (its inference datapipe, T + 1
    rounds): rounds 0 .. T-1 of the measurement record become this
    repository's check values and the noiseless round T becomes the
    final data syndrome; nvidia_input then gives the datapipe's trainX
    bit for bit."""
    lay = nv.surface_layout(D)
    shots = 256
    with nv.nvidia_imports():
        from data.datapipe_stim import QCDataPipePreDecoder_Memory_inference
        torch.manual_seed(0)
        pipe = QCDataPipePreDecoder_Memory_inference(
            distance=D, n_rounds=R, num_samples=shots,
            error_mode="circuit_level_surface_custom", p_error=0.01,
            measure_basis="Z", code_rotation=nv.ORIENTATION)
        ref = torch.stack([pipe[i]["trainX"] for i in range(shots)]).numpy()
        x_names, z_names = nvidia_names(lay, nv.ORIENTATION)
    # NVIDIA measures [X checks, Z checks] per round with reset, so a
    # measurement is the check value
    meas = pipe.meas.numpy()                      # (shots, R, D*D - 1)
    cm = np.zeros((shots, T, lay.n_checks), dtype=np.uint8)
    for i, name in enumerate(x_names + z_names):
        cm[:, :, lay.col[name]] = meas[:, :T, i]
    # final data bits with H_z . dat = Z values of the noiseless round T
    z_last = np.zeros((shots, len(z_names)), dtype=np.uint8)
    for i, name in enumerate(z_names):
        z_last[:, lay.names["Z"].index(name)] = meas[:, T, len(x_names) + i]
    dat = np.asarray(pymatching.Matching(lay.H["Z"]).decode_batch(z_last),
                     dtype=np.uint8)
    assert np.array_equal((dat @ lay.H["Z"].T) % 2, z_last)
    mine = nv.nvidia_input(cm.reshape(shots, -1), dat, T, D)
    assert ref[:, 0].any() and ref[:, 1, T].any()
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
    frame of NVIDIA's PreDecoderMemoryEvalModule (T + 1 rounds). The X
    block of NVIDIA's last round has no detector in this repository and
    is skipped; NVIDIA's boundary block stays zero."""
    lay = nv.surface_layout(D)
    syn, dat = our_samples(256, d=D)
    det = detectors_from_dataset_rotatedSurface(syn, dat, T, D)
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(syn.shape[0], 4, R, D, D, generator=g)
    corr, pre_l = nv.detector_correction(logits.numpy(), T, D)
    mine = det ^ corr
    cfg = types.SimpleNamespace(
        distance=D, enable_fp16=False,
        test=types.SimpleNamespace(meas_basis_test="Z", th_data=0.0,
                                   th_syn=0.0, n_rounds=R),
        data=types.SimpleNamespace(code_rotation=nv.ORIENTATION))
    with nv.nvidia_imports():
        from evaluation.logical_error_rate import (
            PreDecoderMemoryEvalModule, _build_stab_maps)
        x_names, z_names = nvidia_names(lay, nv.ORIENTATION)
        module = PreDecoderMemoryEvalModule(
            _FixedLogits(logits), cfg, _build_stab_maps(D, nv.ORIENTATION),
            torch.device("cpu"))
        out = module(torch.from_numpy(
            to_nvidia_dets(lay, det, x_names, z_names))).numpy()
    assert np.array_equal(pre_l, out[:, 0])
    half = len(z_names)
    expect = to_nvidia_dets(lay, mine, x_names, z_names)
    theirs = out[:, 1:]
    last_x = slice(half + (T - 1) * 2 * half, half + (T - 1) * 2 * half + half)
    keep = np.ones(expect.shape[1], dtype=bool)
    keep[last_x] = False
    assert np.array_equal(expect[:, keep], theirs[:, keep])
    assert not theirs[:, -half:].any()               # boundary block
    assert corr[:, [lay.detector_index(T)[("F", n)] for n in z_names]].any()


def test_negative_logits_reduce_to_mwpm():
    """No correction predicted -> residual = detectors, logical frame 0."""
    syn, dat = our_samples(64)
    logits = -np.ones((syn.shape[0], 4, R, D, D), dtype=np.float32)
    corr, pre_l = nv.detector_correction(logits, T, D)
    assert not corr.any() and not pre_l.any()


@pytest.mark.parametrize("rnd", range(R))
@pytest.mark.parametrize("qubit", range(D * D))
def test_single_x_correction(qubit, rnd):
    """One X correction on one data qubit at one round flips exactly the
    Z detectors of that round whose support holds the qubit (round T =
    the final-Z detectors), and the logical frame iff the qubit is on
    logical Z."""
    lay = nv.surface_layout(D)
    logits = -np.ones((1, 4, R, D, D), dtype=np.float32)
    logits[0, 1, rnd, qubit // D, qubit % D] = 1.0
    corr, pre_l = nv.detector_correction(logits, T, D)
    index = lay.detector_index(T)
    want = np.zeros_like(corr)
    for i, name in enumerate(lay.names["Z"]):
        if lay.H["Z"][i, qubit]:
            key = ("Z", name, rnd) if rnd < T else ("F", name)
            want[0, index[key]] = 1
    assert np.array_equal(corr, want)
    assert int(pre_l[0]) == int(qubit in surface_def(D).LOGICAL_Z_IDX)


@pytest.mark.parametrize("ctype", ["Z", "X"])
def test_single_measurement_correction(ctype):
    """A check-measurement correction at round t flips that check's
    detectors at rounds t and t+1 (the time-like pair), as far as they
    exist."""
    lay = nv.surface_layout(D)
    index = lay.detector_index(T)

    def key(name, t):
        if ctype == "Z":
            return ("Z", name, t) if t < T else ("F", name)
        return ("X", name, t)

    channel = 3 if ctype == "Z" else 2
    for name in lay.names[ctype]:
        r, c = lay.cell[name]
        for rnd in range(R):
            logits = -np.ones((1, 4, R, D, D), dtype=np.float32)
            logits[0, channel, rnd, r, c] = 1.0
            corr, pre_l = nv.detector_correction(logits, T, D)
            want = np.zeros_like(corr)
            for t in (rnd, rnd + 1):
                if t <= T and key(name, t) in index:
                    want[0, index[key(name, t)]] = 1
            assert np.array_equal(corr, want), (name, rnd)
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
    syn0, dat0 = split_rotatedSurface_sample(raw, T, D)
    logits = nv.predecoder_logits(model, nv.nvidia_input(syn0, dat0, T, D))
    corr, pre_l = nv.detector_correction(logits, T, D)
    assert not corr.any() and not pre_l.any()

    syn, dat = our_samples(2000)
    matching = build_matching(T, "X", 0.01, NOISE, "surface", D)
    pred, det, residual, pre_l = nv.nvidia_mwpm_decode(
        model, syn, dat, T, matching, D, return_parts=True)
    y = logical_label_rotatedSurface(dat, D)
    assert pred.shape == y.shape and set(np.unique(pred)) <= {0, 1}
    assert residual.shape == det.shape
