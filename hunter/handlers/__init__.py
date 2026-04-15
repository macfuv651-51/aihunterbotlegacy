"""
handlers/__init__.py
--------------------
Handlers package. Re-exports the handler registration function.
"""

from handlers.message_handler import register_handlers

__all__ = [
    "register_handlers",
]
