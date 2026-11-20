"""Canonical semantic fingerprints for manifests and checkpoint validation."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _schema_type_name(value: Any) -> str:
    """Keep the frozen Goodreads schema's type tokens across this code copy.

    Earlier prepared histories/populations embed module-qualified dataclass
    names in semantic hashes. This is a string-level schema alias, not an
    import from the earlier scientific implementation. Without it, identical
    frozen inputs would falsely acquire new fingerprints in the local namespace.
    """
    module = type(value).__module__
    local = "new_dataset_experiment_mitigation.engine."
    if module.startswith(local):
        module = "new_dataset_experiment.engine." + module[len(local):]
    return f"{module}.{type(value).__qualname__}"


def _canonical(value: Any) -> Any:
    """Return a deterministic, JSON-compatible semantic representation."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("semantic state cannot contain NaN or infinity")
        return value
    if isinstance(value, Enum):
        return {
            "__enum__": _schema_type_name(value),
            "value": _canonical(value.value),
        }
    if isinstance(value, Path):
        return {"__path__": value.as_posix()}
    if isinstance(value, bytes):
        return {
            "__bytes_sha256__": hashlib.sha256(value).hexdigest(),
            "length": len(value),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": _schema_type_name(value),
            "fields": {
                field.name: _canonical(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, Mapping):
        entries = [(_canonical(key), _canonical(item)) for key, item in value.items()]
        entries.sort(
            key=lambda pair: json.dumps(
                pair[0], sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        )
        return {"__mapping__": entries}
    if isinstance(value, (tuple, list)):
        return {
            "__sequence_type__": type(value).__name__,
            "items": [_canonical(item) for item in value],
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item) for item in value]
        items.sort(
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        )
        return {"__set_type__": type(value).__name__, "items": items}

    # NumPy and PyTorch are optional at import time.  Duck-typing here keeps
    # this utility usable for manifests that do not need either dependency.
    module = type(value).__module__
    if module.startswith("numpy") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        contiguous = value.copy(order="C")
        payload = contiguous.tobytes(order="C")
        return {
            "__ndarray__": True,
            "dtype": str(contiguous.dtype),
            "shape": tuple(int(dimension) for dimension in contiguous.shape),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if module.startswith("torch") and hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        array = tensor.numpy()
        return {
            "__tensor__": True,
            "dtype": str(tensor.dtype),
            "shape": tuple(int(dimension) for dimension in tensor.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    semantic_payload = getattr(value, "semantic_payload", None)
    if callable(semantic_payload):
        return _canonical(semantic_payload())
    raise TypeError(f"unsupported semantic-state value: {type(value)!r}")


def fingerprint_semantic_state(value: Any, schema_version: str) -> str:
    """Hash canonical, versioned state without timestamps or object identity."""

    if not isinstance(schema_version, str) or not schema_version.strip():
        raise ValueError("schema_version must be a non-blank string")
    payload = {
        "schema_version": schema_version,
        "state": _canonical(value),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
