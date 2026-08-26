from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

_INFERENCE_GUARD = threading.local()


def is_inference_guard_active() -> bool:
    return bool(getattr(_INFERENCE_GUARD, "active", False))


def record_auxiliary_head_call(head_name: str) -> None:
    if is_inference_guard_active():
        raise RuntimeError(
            f"Action-only inference must not call {head_name}. "
            "FutureLatentHead / InverseActionHead are train-only auxiliary modules."
        )


@contextmanager
def inference_guard() -> Iterator[None]:
    """Raise if auxiliary heads are invoked during action-only eval."""
    prev = getattr(_INFERENCE_GUARD, "active", False)
    _INFERENCE_GUARD.active = True
    try:
        yield
    finally:
        _INFERENCE_GUARD.active = prev
