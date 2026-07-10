from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

_AUXILIARY_HEAD_ACTIVE = False


def record_auxiliary_head_call(name: str) -> None:
    global _AUXILIARY_HEAD_ACTIVE
    _AUXILIARY_HEAD_ACTIVE = True


@contextmanager
def inference_guard() -> Generator[None, None, None]:
    """Block auxiliary head usage during sim inference."""
    global _AUXILIARY_HEAD_ACTIVE
    prev = _AUXILIARY_HEAD_ACTIVE
    _AUXILIARY_HEAD_ACTIVE = False
    try:
        yield
    finally:
        _AUXILIARY_HEAD_ACTIVE = prev


def auxiliary_heads_allowed() -> bool:
    return _AUXILIARY_HEAD_ACTIVE
