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
    "gcn": ("model.gcn", "GCNDecoder"),
    "gcnii": ("model.gcnii", "GCNIIDecoder"),
    "gat": ("model.gat", "GATDecoder"),
    "appnp": ("model.appnp", "APPNPDecoder"),
    "gnn": ("model.gnn", "GNNDecoder"),
    "graph_transformer": ("model.graph_transformer",
                          "GraphTransformerDecoder"),
    "graph_mamba": ("model.graph_mamba", "GraphMambaDecoder"),
}
MODEL_NAMES = list(MODEL_REGISTRY)


def model_prefix(model_name):
    """Result / checkpoint file-name prefix of a model: the registry name
    upper-cased without underscores (graph_mamba -> GRAPHMAMBA), so the
    prefix is the part of the file name before the first underscore."""
    return model_name.replace("_", "").upper()


def model_from_prefix(prefix):
    """Inverse of model_prefix; None for an unknown prefix."""
    for name in MODEL_REGISTRY:
        if model_prefix(name) == prefix.upper():
            return name
    return None

# per-(code, distance) model-facing constants: number of per-qubit label
# bits and the (H, W) of the syndrome tensor planes (None until the
# code's tensor layout exists). Read through code_spec().
CODE_SPECS = {
    ("heavyhex", 33): {"num_qubits": 17, "grid": (4, 5)},
    ("surface", 3): {"num_qubits": 9, "grid": (4, 4)},
    ("surface", 5): {"num_qubits": 25, "grid": (6, 6)},
    ("color", 3): {"num_qubits": 7, "grid": (2, 2)},
    ("color", 5): {"num_qubits": 19, "grid": (5, 4)},
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
