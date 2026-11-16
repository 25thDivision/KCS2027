"""Model registry and code / distance axes."""
import importlib
import inspect

import pytest

from dataset_generation import (
    CODES, DISTANCES, IMPLEMENTED, distance_tag, is_implemented,
    normalize_distance, require_implemented)
from model import (
    CODE_SPECS, MODEL_NAMES, MODEL_REGISTRY, code_spec, get_model_class,
    get_model_module)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_registry_entry_resolves(name):
    module_path, class_name = MODEL_REGISTRY[name]
    mod = importlib.import_module(module_path)
    assert get_model_module(name) is mod
    cls = get_model_class(name)
    assert inspect.isclass(cls) and cls.__name__ == class_name
    # module-level interface shared by every model
    assert callable(mod.compute_loss)
    assert callable(mod.prepare_features)
    params = inspect.signature(cls.__init__).parameters
    for arg in ("in_channels", "num_qubits", "code", "distance"):
        assert arg in params, f"{class_name}.__init__ lacks '{arg}'"


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        get_model_module("no_such_model")


def test_codes():
    assert CODES == ("heavyhex", "surface", "color")
    assert set(DISTANCES) == set(CODES)


@pytest.mark.parametrize("code, distance, tag", [
    ("heavyhex", 3, "d3"), ("heavyhex", 33, "d3"),
    ("heavyhex", 35, "d35"), ("heavyhex", 53, "d53"),
    ("surface", 3, "d3"), ("surface", 5, "d5"),
    ("color", 3, "d3"), ("color", 5, "d5"),
])
def test_distance_tag(code, distance, tag):
    assert distance_tag(code, distance) == tag


def test_heavyhex_3_is_33():
    assert normalize_distance("heavyhex", 3) == 33
    assert normalize_distance("heavyhex", "35") == 35


@pytest.mark.parametrize("code, distance", [
    ("heavyhex", 5), ("surface", 33), ("color", 7), ("nope", 3)])
def test_invalid_distance_rejected(code, distance):
    with pytest.raises(ValueError):
        normalize_distance(code, distance)


def test_implemented_pairs_are_valid():
    for code, distance in IMPLEMENTED:
        assert distance in DISTANCES[code]
        assert is_implemented(code, distance)
        require_implemented(code, distance)


def test_existing_paths_keep_working():
    # heavyhex (3,3) and surface d=3 predate the distance axis
    assert is_implemented("heavyhex", 3)
    assert is_implemented("surface", 3)
    assert code_spec("heavyhex", 3) == {"num_qubits": 17, "grid": (4, 5)}
    assert code_spec("surface", 3) == {"num_qubits": 9, "grid": (4, 4)}


def test_color_specs():
    assert code_spec("color", 3)["num_qubits"] == 7
    assert code_spec("color", 5)["num_qubits"] == 19


def test_code_specs_keys_are_normalized():
    for code, distance in CODE_SPECS:
        assert normalize_distance(code, distance) == distance


def test_unimplemented_pair_fails_early():
    for code in CODES:
        for distance in DISTANCES[code]:
            if (code, distance) in IMPLEMENTED:
                continue
            with pytest.raises(NotImplementedError):
                require_implemented(code, distance)


def test_dataset_file_name_unchanged_for_default_distance(tmp_path):
    from model.data import npz_path
    noise = "realistic/dp0.001_mf0.01_rf0.01_gd0.008"
    for code in ("heavyhex", "surface"):
        p = npz_path(tmp_path, code, noise, "train", 3, 0.005, "X")
        assert p.name == "train_d3_c3_p0.005_X.npz"
        assert p.parent.parent.name == code
