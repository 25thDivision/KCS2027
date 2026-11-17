"""Hardware providers: one class per vendor / execution route, chosen by
the backend name prefix (see base.py for the interface)."""
import importlib

# backend-name prefix -> (module, class)
PROVIDERS = {
    "ibm_": ("hardware.providers.ibm", "IBMProvider"),
    "ionq_": ("hardware.providers.ionq", "IonQProvider"),
    "aer_": ("hardware.providers.aer_calibrated", "AerCalibratedProvider"),
}


def provider_name(backend):
    """ibm / ionq / aer of a backend name; ValueError when unknown."""
    for prefix in PROVIDERS:
        if str(backend).startswith(prefix):
            return prefix.rstrip("_")
    raise ValueError(
        f"no provider for backend '{backend}' (known prefixes: "
        f"{sorted(PROVIDERS)})")


def get_provider(backend):
    """Provider instance of a backend name."""
    prefix = provider_name(backend) + "_"
    module, cls = PROVIDERS[prefix]
    return getattr(importlib.import_module(module), cls)(backend)
