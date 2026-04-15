"""
bot/handlers/__init__.py
------------------------
Registers all handler routers with the Dispatcher.

Import register_all_routers and call it once during bot startup.
"""

from aiogram import Dispatcher

from bot.handlers.channels import router as channels_router
from bot.handlers.confirm import router as confirm_router
from bot.handlers.control import router as control_router
from bot.handlers.products import router as products_router
from bot.handlers.start import router as start_router


def register_all_routers(dp: Dispatcher) -> None:
    """
    Include every handler router into the dispatcher.

    Router order matters — start_router is first so /start is matched
    before the text-based navigation handlers.

    Args:
        dp: The main Dispatcher instance.
    """
    dp.include_router(start_router)
    dp.include_router(confirm_router)
    dp.include_router(channels_router)
    dp.include_router(control_router)
    dp.include_router(products_router)
