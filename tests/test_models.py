"""Common contract of every registered decoder model.

For each model x (heavyhex (3,3), surface d=3), cycles=3:
  (a) the class builds from model.code_spec
  (b) forward on a prepare_features tensor returns (B, num_qubits), (B, 1)
  (c) compute_loss returns three scalars, total = logical + aux * qubit
  (d) one backward step puts a gradient on every parameter
  (e) eval-mode output is identical for the same input
"""
import inspect

import pytest
import torch

from model import (
    MODEL_NAMES, code_spec, get_model_class, get_model_module, model_prefix,
    model_from_prefix)
from model.common import dual_head_loss

BATCH = 8
CYCLES = 3
CASES = [("heavyhex", 3), ("surface", 3), ("surface", 5), ("color", 3),
         ("color", 5)]
NEEDS_CUDA = {"graph_mamba"}

PARAMS = [pytest.param(name, code, distance, id=f"{name}-{code}{distance}")
          for name in MODEL_NAMES for code, distance in CASES]


def _device(name):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if name in NEEDS_CUDA:
        pytest.skip(f"{name} needs CUDA")
    return torch.device("cpu")


def build_model(name, code, distance, device):
    spec = code_spec(code, distance)
    torch.manual_seed(0)
    cls = get_model_class(name)
    model = cls(in_channels=2 * CYCLES, num_qubits=spec["num_qubits"],
                code=code, distance=distance)
    return model.to(device)


def make_batch(mod, code, distance, device):
    """Random uint8 dataset tensor + final bits, through prepare_features,
    plus labels shaped like the npz ones (uint8)."""
    spec = code_spec(code, distance)
    h, w = spec["grid"]
    nq = spec["num_qubits"]
    g = torch.Generator().manual_seed(1)
    feats = torch.randint(0, 2, (BATCH, 2 * CYCLES, h, w), generator=g,
                          dtype=torch.uint8).to(device)
    bits = torch.randint(0, 2, (BATCH, nq), generator=g,
                         dtype=torch.uint8).to(device)
    y_logical = torch.randint(0, 2, (BATCH,), generator=g,
                              dtype=torch.uint8).to(device)
    x = mod.prepare_features(feats, bits, code, distance)
    return x, bits, y_logical


def check_model(mod, model, code, distance, device):
    """The (b) - (e) checks; shared with any model object."""
    nq = code_spec(code, distance)["num_qubits"]
    x, y_qubit, y_logical = make_batch(mod, code, distance, device)
    assert x.shape[1] == 2 * CYCLES + 1 and x.dtype == torch.uint8

    # (b) forward shapes
    model.train()
    out = model(x)
    assert isinstance(out, tuple) and len(out) == 2
    ql, ll = out
    assert tuple(ql.shape) == (BATCH, nq)
    assert tuple(ll.shape) == (BATCH, 1)
    assert ql.is_floating_point() and ll.is_floating_point()
    assert torch.isfinite(ql).all() and torch.isfinite(ll).all()

    # (c) loss: three scalars, total = logical + aux_weight * qubit
    for aux, pos_weight in ((0.5, None),
                            (0.3, torch.full((nq,), 2.0, device=device))):
        total, l_log, l_q = mod.compute_loss(ql, ll, y_qubit, y_logical,
                                             aux, pos_weight)
        for v in (total, l_log, l_q):
            assert v.dim() == 0 and torch.isfinite(v)
        assert abs(total.item() - (l_log.item() + aux * l_q.item())) < 1e-6

    # (d) one backward step reaches every parameter
    total, _, _ = mod.compute_loss(ql, ll, y_qubit, y_logical, 0.5, None)
    model.zero_grad()
    total.backward()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert params, "model has no trainable parameters"
    missing = [n for n, p in params if p.grad is None]
    assert not missing, f"no gradient on: {missing}"

    # (e) eval mode is deterministic
    model.eval()
    with torch.no_grad():
        q1, l1 = model(x)
        q2, l2 = model(x)
    assert torch.allclose(q1, q2, rtol=0, atol=1e-6)
    assert torch.allclose(l1, l2, rtol=0, atol=1e-6)


@pytest.mark.parametrize("name, code, distance", PARAMS)
def test_model_contract(name, code, distance):
    device = _device(name)
    mod = get_model_module(name)
    model = build_model(name, code, distance, device)      # (a)
    check_model(mod, model, code, distance, device)


@pytest.mark.parametrize("name, code, distance", PARAMS)
def test_state_dict_roundtrip(name, code, distance):
    """A checkpoint loads into a freshly built model and reproduces the
    outputs (train.py saves state_dict, hardware/run_hw.py reloads it)."""
    device = _device(name)
    mod = get_model_module(name)
    a = build_model(name, code, distance, device).eval()
    torch.manual_seed(123)
    b = get_model_class(name)(
        in_channels=2 * CYCLES,
        num_qubits=code_spec(code, distance)["num_qubits"],
        code=code, distance=distance).to(device).eval()
    b.load_state_dict(a.state_dict())
    x, _, _ = make_batch(mod, code, distance, device)
    with torch.no_grad():
        qa, la = a(x)
        qb, lb = b(x)
    assert torch.allclose(qa, qb, rtol=0, atol=1e-6)
    assert torch.allclose(la, lb, rtol=0, atol=1e-6)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_module_is_pure_torch(name):
    src = inspect.getsource(get_model_module(name))
    assert "import torch_geometric" not in src
    assert "from torch_geometric" not in src


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_prefix_roundtrip(name):
    prefix = model_prefix(name)
    assert "_" not in prefix and prefix == prefix.upper()
    assert model_from_prefix(prefix) == name


def test_unknown_prefix():
    assert model_from_prefix("NOPE") is None


def test_dual_head_loss_known_values():
    # logits 0 -> BCE = ln 2 for any label
    ql = torch.zeros(4, 3)
    ll = torch.zeros(4, 1)
    yq = torch.tensor([[0, 1, 0]] * 4, dtype=torch.uint8)
    yl = torch.tensor([0, 1, 1, 0], dtype=torch.uint8)
    total, l_log, l_q = dual_head_loss(ql, ll, yq, yl, aux_weight=0.5)
    ln2 = 0.6931471805599453
    assert abs(l_log.item() - ln2) < 1e-6
    assert abs(l_q.item() - ln2) < 1e-6
    assert abs(total.item() - 1.5 * ln2) < 1e-6
    # pos_weight scales only the positive-label terms of the qubit head
    pw = torch.tensor([1.0, 3.0, 1.0])
    _, l_log2, l_q2 = dual_head_loss(ql, ll, yq, yl, 0.5, pw)
    assert abs(l_log2.item() - ln2) < 1e-6
    assert abs(l_q2.item() - ln2 * (1 + 3 + 1) / 3) < 1e-6
