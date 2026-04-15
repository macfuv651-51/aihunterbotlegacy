"""
bot/handlers/control.py
-----------------------
Handles the Telethon recognition toggle and the full status overview screen.
"""

from aiogram import F, Router
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from bot.keyboards.main_menu import BTN_STATUS, BTN_TELETHON
from data.app_state import (
    get_chats,
    is_telethon_enabled,
    set_telethon_enabled,
)
from utils.logger import get_logger

logger = get_logger(__name__)
router = Router()

# Callback identifier for the toggle button
_CB_TOGGLE = "telethon_toggle"


def _telethon_keyboard() -> InlineKeyboardMarkup:
    """
    Build a single-button inline keyboard to toggle Telethon recognition.

    The button label reflects the current state so the admin always knows
    what will happen when they press it.

    Returns:
        InlineKeyboardMarkup with one toggle button.
    """
    enabled = is_telethon_enabled()
    label = (
        "🟢 Распознавание включено — нажмите, чтобы выключить"
        if enabled
        else "🔴 Распознавание выключено — нажмите, чтобы включить"
    )
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=label, callback_data=_CB_TOGGLE)]
        ]
    )


@router.message(F.text == BTN_TELETHON)
async def show_telethon_panel(message: Message) -> None:
    """
    Show the current Telethon recognition state with a toggle button.

    Args:
        message: Incoming message with text '🔌 Telethon'.
    """
    enabled = is_telethon_enabled()
    status = "🟢 <b>Включено</b>" if enabled else "🔴 <b>Выключено</b>"
    await message.answer(
        f"<b>🔌 Распознавание сообщений Telethon</b>\n\n"
        f"Текущий статус: {status}\n\n"
        "Когда выключено, <b>ключевые слова не распознаются</b> "
        "и уведомления не отправляются, даже если чаты активны.",
        reply_markup=_telethon_keyboard(),
    )


@router.callback_query(F.data == _CB_TOGGLE)
async def toggle_telethon(callback: CallbackQuery) -> None:
    """
    Toggle Telethon recognition on or off and update the inline button.

    Args:
        callback: Callback query from the toggle button.
    """
    current = is_telethon_enabled()
    set_telethon_enabled(not current)

    new_status = "включено ✅" if not current else "выключено ❌"
    logger.info("Admin toggled Telethon recognition → %s.", new_status)
    await callback.answer(f"Распознавание Telethon теперь {new_status}.")

    # Update button label in-place
    await callback.message.edit_reply_markup(reply_markup=_telethon_keyboard())


@router.message(F.text == BTN_STATUS)
async def show_status(message: Message) -> None:
    """
    Display a full overview: Telethon state and the complete chat list.

    Args:
        message: Incoming message with text '📊 Status'.
    """
    enabled = is_telethon_enabled()
    chats = get_chats()

    telethon_line = "🟢 Включено" if enabled else "🔴 Выключено"

    if chats:
        chat_lines = "\n".join(
            f"  {'✅' if c.get('enabled', True) else '❌'}  "
            f"{c.get('title') or c['id']}  <code>({c['id']})</code>"
            for c in chats
        )
    else:
        chat_lines = "  <i>Пока нет добавленных чатов</i>"

    await message.answer(
        f"<b>📊 Текущий статус</b>\n\n"
        f"<b>Распознавание Telethon:</b> {telethon_line}\n\n"
        f"<b>Отслеживаемые чаты ({len(chats)}):</b>\n{chat_lines}"
    )
