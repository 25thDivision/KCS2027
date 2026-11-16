#!/usr/bin/env python3
"""
NVIDIA Ising pre-decoder on this repository's rotated surface code
==================================================================
Wraps the pre-decoder of the NVIDIA Ising-Decoding clone
(./Ising-decoding, see docs/nvidia_api_notes.md) for the surface path:

  check values + final data bits
    -> nvidia_input           (B, 4, T, D, D) float32 model input
    -> predecoder_logits      (B, 4, T, D, D) raw logits
    -> detector_correction    correction in THIS repository's detector
                              order + the pre-decoder's own logical flip
    -> residual detectors = detectors ^ correction
    -> baseline/mwpm.py Matching on the residual
    -> logical flip = MWPM prediction ^ pre-decoder logical flip

Lattice: with data index = row * D + col, row = (y - 1) / 2, col =
(x - 1) / 2 of circuits/rotatedSurface, this repository's code is cell
for cell NVIDIA's orientation O4 (internal name ZH): same X / Z
stabilizer supports, logical Z on column 0. No relabeling is applied.

Input channels (memory-Z basis, T = num_cycles):
  0 x_type     X detectors on the D x D grid; cycles 0 and T-1 are zero
               (the X detector of cycle t is check[t] ^ check[t-1])
  1 z_type     Z detectors; cycle 0 is the check value itself
  2 x_present  1.0 (weight-4) / 0.5 (weight-2) at the X anchor cells;
               cycles 0 and T-1 are zero
  3 z_present  same for Z, every cycle
A stabilizer sits at its anchor data cell:
  X  weight-4 -> top-left;  horizontal pair -> left;  vertical pair -> top
  Z  weight-4 -> top-right; horizontal pair -> right; vertical pair -> top
("top" = smallest row, "left" = smallest col.)

Output channels (thresholded at 0):
  0 Z corrections on data   1 X corrections on data
  2 X-check measurement corrections   3 Z-check measurement corrections

The final-Z detectors are not shown to the network and are never
corrected: a data correction at cycle t <= T-1 flips the last check
value and the final data parity together.
"""
import contextlib
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_generation import require_implemented  # noqa: E402

NVIDIA_CODE = _ROOT / "Ising-decoding" / "code"
ORIENTATION = "ZH"                    # NVIDIA public alias O4


def _surface_modules(distance):
    """(circuit module, stim module helpers) of the surface path."""
    require_implemented("surface", distance)
    from circuits.rotatedSurface import rotatedSurface3 as circ
    from dataset_generation.rotatedSurface3_stim import (
        detectors_from_dataset_rotatedSurface3 as det_fn)
    return circ, det_fn


def _anchor(ctype, cells):
    """Support cells [(row, col), ...] -> anchor cell (module docstring)."""
    if len(cells) == 2:
        horizontal = cells[0][0] == cells[1][0]
        if ctype == "X":
            key = (lambda rc: rc[1]) if horizontal else (lambda rc: rc[0])
        else:
            key = (lambda rc: -rc[1]) if horizontal else (lambda rc: rc[0])
    elif ctype == "X":
        key = lambda rc: (rc[0], rc[1])                  # noqa: E731
    else:
        key = lambda rc: (rc[0], -rc[1])                 # noqa: E731
    return min(cells, key=key)


class SurfaceLayout:
    """Grid placement and parity checks of the surface code, derived
    from the lattice coordinates of circuits/rotatedSurface.

    names[t]      check names of type t ("Z" / "X") in CYCLE_ORDER order
    col[name]     column of the check inside one cycle of a check matrix
    cell[name]    anchor (row, col) on the D x D grid
    weight[name]  1.0 (weight-4) or 0.5 (weight-2)
    H[t]          (len(names[t]), D*D) uint8 parity-check matrix
    logical_z     data indices of logical Z"""

    def __init__(self, distance=3):
        circ, _ = _surface_modules(distance)
        self.distance = D = circ.DISTANCE
        self.n_checks = circ.N_CHECKS
        self.num_data = circ.NUM_DATA
        self.cycle_order = list(circ.CYCLE_ORDER)
        self.names = {"Z": list(circ.Z_NAMES), "X": list(circ.X_NAMES)}
        self.col = {n: j for j, n in enumerate(circ.CYCLE_ORDER)}
        self.logical_z = list(circ.LOGICAL_Z_IDX)
        self.cell, self.weight = {}, {}
        self.H = {t: np.zeros((len(self.names[t]), D * D), dtype=np.uint8)
                  for t in ("Z", "X")}
        for name in circ.CYCLE_ORDER:
            ctype, support = circ.CHECK_DEFS[name][0], circ.CHECK_DEFS[name][1]
            cells = [((y - 1) // 2, (x - 1) // 2) for x, y in support]
            for (x, y), (r, c) in zip(support, cells):
                assert circ.DIDX[(x, y)] == r * D + c, "data index layout"
                self.H[ctype][self.names[ctype].index(name), r * D + c] = 1
            self.cell[name] = _anchor(ctype, cells)
            self.weight[name] = 0.5 if len(cells) == 2 else 1.0
        for t in ("Z", "X"):
            assert len({self.cell[n] for n in self.names[t]}) == \
                len(self.names[t]), "anchor cells collide"

    def num_detectors(self, num_cycles):
        nz, nx = len(self.names["Z"]), len(self.names["X"])
        return nz * num_cycles + nx * (num_cycles - 1) + nz

    def detector_index(self, num_cycles):
        """{("Z"|"X", name, cycle) | ("F", name): column} in the detector
        order of dataset_generation/rotatedSurface3_stim._append_detectors:
        per cycle over CYCLE_ORDER (Z every cycle, X from cycle 1), then
        the final-Z detectors."""
        idx, k = {}, 0
        for cyc in range(num_cycles):
            for name in self.cycle_order:
                if name in self.names["Z"]:
                    idx[("Z", name, cyc)] = k
                    k += 1
                elif cyc >= 1:
                    idx[("X", name, cyc)] = k
                    k += 1
        for name in self.names["Z"]:
            idx[("F", name)] = k
            k += 1
        assert k == self.num_detectors(num_cycles)
        return idx


_LAYOUTS = {}


def surface_layout(distance=3):
    if distance not in _LAYOUTS:
        _LAYOUTS[distance] = SurfaceLayout(distance)
    return _LAYOUTS[distance]


# ==================================================================
# model
# ==================================================================
# top-level packages of Ising-decoding/code; `model` and `evaluation`
# collide with this repository's packages of the same name
_NVIDIA_PACKAGES = ("model", "evaluation", "data", "export", "workflows",
                    "qec", "training")


@contextlib.contextmanager
def nvidia_imports():
    """Import from Ising-decoding/code inside this block.

    The same-named modules of this repository are set aside while the
    block runs and put back afterwards; the NVIDIA modules are dropped
    from sys.modules on exit (objects created from them stay valid), so
    every NVIDIA import, including the lazy ones inside NVIDIA functions,
    has to happen inside the block."""
    def ours(name):
        return name.split(".")[0] in _NVIDIA_PACKAGES

    if not NVIDIA_CODE.exists():
        raise FileNotFoundError(
            f"{NVIDIA_CODE} not found: clone NVIDIA/Ising-Decoding into "
            f"./Ising-decoding")
    saved = {k: v for k, v in sys.modules.items() if ours(k)}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(NVIDIA_CODE))
    try:
        yield
    finally:
        sys.path.remove(str(NVIDIA_CODE))
        for k in [k for k in sys.modules if ours(k)]:
            del sys.modules[k]
        sys.modules.update(saved)


def load_predecoder(checkpoint, device="cpu"):
    """Load a public pre-decoder checkpoint (.safetensors) through the
    NVIDIA loader. The fp16 weights are cast to fp32 so the model also
    runs on CPU. Returns the model in eval mode on `device`."""
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} not found (see docs/nvidia_api_notes.md)")
    with nvidia_imports():
        from export.safetensors_utils import load_safetensors
        model, _ = load_safetensors(str(checkpoint), device=str(device))
    return model.float().eval()


def predecoder_logits(model, x, batch_size=8192):
    """(B, 4, T, D, D) float32 numpy input -> logits, same shape."""
    import torch
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i:i + batch_size]).to(device)
            out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros_like(x)


# ==================================================================
# input / output conversion
# ==================================================================
def nvidia_input(check_mat, num_cycles, distance=3):
    """(shots, N_CHECKS * C) check values -> (shots, 4, C, D, D) float32."""
    lay = surface_layout(distance)
    D, T = lay.distance, num_cycles
    check_mat = np.asarray(check_mat, dtype=np.uint8)
    n = check_mat.shape[0]
    cm = check_mat.reshape(n, T, lay.n_checks)
    x = np.zeros((n, 4, T, D, D), dtype=np.float32)
    for name in lay.names["Z"]:
        r, c = lay.cell[name]
        v = cm[:, :, lay.col[name]]
        det = v.copy()
        det[:, 1:] ^= v[:, :-1]
        x[:, 1, :, r, c] = det
        x[:, 3, :, r, c] = lay.weight[name]
    for name in lay.names["X"]:
        r, c = lay.cell[name]
        v = cm[:, :, lay.col[name]]
        if T > 2:
            x[:, 0, 1:T - 1, r, c] = v[:, 1:T - 1] ^ v[:, :T - 2]
            x[:, 2, 1:T - 1, r, c] = lay.weight[name]
    return x


def detector_correction(logits, num_cycles, distance=3,
                        th_data=0.0, th_syn=0.0):
    """Pre-decoder logits -> (correction, pre_logical).

    correction   (shots, num_detectors) uint8 in this repository's
                 detector order; XOR it into the detectors to get the
                 residual. For check type s, cycle t:
                   corr[t] = syn[t] ^ syn[t-1] ^ H_s . data[t]   (syn[-1]=0)
                 with data = X corrections for Z checks and Z corrections
                 for X checks. Final-Z detectors stay 0.
    pre_logical  (shots,) uint8: parity of the X corrections on the
                 logical-Z support over all cycles."""
    lay = surface_layout(distance)
    D, T = lay.distance, num_cycles
    logits = np.asarray(logits)
    n = logits.shape[0]
    data = {"X": (logits[:, 0] >= th_data),          # Z errors -> X checks
            "Z": (logits[:, 1] >= th_data)}          # X errors -> Z checks
    syn = {"X": (logits[:, 2] >= th_syn), "Z": (logits[:, 3] >= th_syn)}
    nn = {}
    for t in ("Z", "X"):
        flat = data[t].reshape(n, T, D * D).astype(np.int64)
        induced = (flat @ lay.H[t].T.astype(np.int64)) & 1    # (n, T, S)
        pred = np.stack([syn[t][:, :, r, c] for r, c in
                         (lay.cell[nm] for nm in lay.names[t])],
                        axis=2).astype(np.int64)
        corr = pred ^ induced
        corr[:, 1:] ^= pred[:, :-1]
        nn[t] = corr.astype(np.uint8)
    index = lay.detector_index(T)
    out = np.zeros((n, lay.num_detectors(T)), dtype=np.uint8)
    for key, k in index.items():
        if key[0] == "F":
            continue
        t, name, cyc = key
        out[:, k] = nn[t][:, cyc, lay.names[t].index(name)]
    x_flat = data["Z"].reshape(n, T, D * D)
    pre_logical = (x_flat[:, :, lay.logical_z].sum(axis=(1, 2)) & 1
                   ).astype(np.uint8)
    return out, pre_logical


def nvidia_mwpm_decode(model, check_mat, dat, num_cycles, matching,
                       distance=3, return_parts=False):
    """Pre-decoder + MWPM on the residual -> predicted logical flips.

    check_mat  (shots, N_CHECKS * C) check values
    dat        (shots, num_data) final data bits
    matching   pymatching.Matching of baseline/mwpm.build_matching
    return_parts=True also returns (detectors, residual, pre_logical)."""
    _, det_fn = _surface_modules(distance)
    det = det_fn(np.asarray(check_mat, dtype=np.uint8),
                 np.asarray(dat, dtype=np.uint8), num_cycles)
    logits = predecoder_logits(model, nvidia_input(check_mat, num_cycles,
                                                   distance))
    corr, pre_logical = detector_correction(logits, num_cycles, distance)
    residual = det ^ corr
    pred = np.asarray(matching.decode_batch(residual), dtype=np.uint8)
    pred = pred.reshape(pred.shape[0], -1)[:, 0] ^ pre_logical
    if return_parts:
        return pred, det, residual, pre_logical
    return pred
