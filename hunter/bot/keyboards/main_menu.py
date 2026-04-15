"""
bot/keyboards/main_menu.py
--------------------------
Persistent bottom navigation ReplyKeyboard shown to every authorized user.

Uses ReplyKeyboardMarkup so the buttons sit in the keyboard area,
not attached to a message (unlike InlineKeyboardMarkup).
"""

from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

# Button label constants — imported by handlers to match incoming text
BTN_CHANNELS: str = "📋 Каналы"
BTN_TELETHON: str = "🔌 Распознавание"
BTN_STATUS: str = "📊 Статус"
BTN_PRODUCTS: str = "📦 Наши товары"


def get_main_menu() -> ReplyKeyboardMarkup:
    """
    Build and return the persistent bottom navigation keyboard.

    Layout:
        Row 1 — [ 📋 Channels ]  [ 🔌 Telethon ]
        Row 2 — [ 📊 Status   ]  [ 📦 Products ]

    Returns:
        ReplyKeyboardMarkup that replaces the system keyboard.
    """
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text=BTN_CHANNELS),
                KeyboardButton(text=BTN_TELETHON),
            ],
            [
                KeyboardButton(text=BTN_STATUS),
                KeyboardButton(text=BTN_PRODUCTS),
            ],
        ],
        resize_keyboard=True,   # shrink to fit buttons
        persistent=True,        # keep keyboard open between messages
    )
