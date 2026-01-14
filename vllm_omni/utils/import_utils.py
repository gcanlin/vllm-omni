"""
Import utilities for dynamic class loading.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import types


def import_pynvml() -> "types.ModuleType":
    """
    Import pynvml module.

    Try nvidia-ml-py first (official), then fall back to pynvml.
    """
    try:
        import nvidia_ml_py as pynvml

        return pynvml
    except ImportError:
        pass

    import pynvml

    return pynvml


def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve a qualified name to an object.

    Example: "vllm_omni.worker.gpu_ar_worker.GPUARWorker" -> GPUARWorker class

    Args:
        qualname: Fully qualified name (module.submodule.ClassName)

    Returns:
        The resolved object (class, function, etc.)

    Raises:
        ValueError: If qualname is invalid
        ImportError: If module cannot be imported
        AttributeError: If object not found in module
    """
    module_name, _, obj_name = qualname.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid qualified name: {qualname}")
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)
