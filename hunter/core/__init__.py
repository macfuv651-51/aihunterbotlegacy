"""
core/__init__.py
----------------
Core package. Re-exports the public API.
"""

from core.client import build_client
from core.session import authorize

__all__ = [
    "build_client",
    "authorize",
]
