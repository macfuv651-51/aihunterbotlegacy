"""
bot/middlewares/__init__.py
---------------------------
Middlewares sub-package.
"""

from bot.middlewares.auth import AdminOnlyMiddleware

__all__ = ["AdminOnlyMiddleware"]
