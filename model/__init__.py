"""Model registry — single place that resolves --model names.

Every model module exposes the same interface: a decoder class
`<Class>(in_channels=2*cycles, num_qubits, code, distance, ...)` with
`forward(x) -> (qubit_logits, logical_logits)` plus module-level
`compute_loss(...) -> (total, loss_logical, loss_qubit)` and
`prepare_features(features, final_bits, code, distance)`.
`get_model_module` / `get_model_class` resolve a registry name for
train.py and hardware/run_hw.py.
"""
import importlib

from dataset_generation import normalize_distance

# name -> (module path, decoder class name)
MODEL_REGISTRY = {
    "cnn": ("model.cnn", "CNNDecoder"),
    "gnn": ("model.gnn", "GNNDecoder"),
}
MODEL_NAMES = sorted(MODEL_REGISTRY)

# per-(code, distance) model-facing constants: number of per-qubit label
# bits and the (H, W) of the syndrome tensor planes (None until the
# code's tensor layout exists). Read through code_spec().
CODE_SPECS = {
    ("heavyhex", 33): {"num_qubits": 17, "grid": (4, 5)},
    ("surface", 3): {"num_qubits": 9, "grid": (4, 4)},
    ("color", 3): {"num_qubits": 7, "grid": None},
    ("color", 5): {"num_qubits": 19, "grid": None},
}


def code_spec(code, distance=3):
    """Model-facing constants of one (code, distance)."""
    key = (code, normalize_distance(code, distance))
    if key not in CODE_SPECS:
        raise NotImplementedError(
            f"no model spec for --code {code} --distance {distance} yet "
            f"(available: {sorted(CODE_SPECS)})")
    return CODE_SPECS[key]


def get_model_module(model_name):
    """Import and return the module of a model."""
    try:
        module_path, _ = MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"unknown model '{model_name}' "
                         f"(available: {MODEL_NAMES})")
    return importlib.import_module(module_path)


def get_model_class(model_name):
    """Return the decoder class (e.g. CNNDecoder) of a model."""
    mod = get_model_module(model_name)
    return getattr(mod, MODEL_REGISTRY[model_name][1])
