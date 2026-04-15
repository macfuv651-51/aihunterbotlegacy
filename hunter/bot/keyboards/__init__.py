"""
bot/keyboards/__init__.py
-------------------------
Keyboards sub-package.
"""

from bot.keyboards.main_menu import BTN_CHANNELS, BTN_STATUS, BTN_TELETHON, get_main_menu

__all__ = [
    "get_main_menu",
    "BTN_CHANNELS",
    "BTN_TELETHON",
    "BTN_STATUS",
]
