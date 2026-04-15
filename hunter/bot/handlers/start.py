"""
bot/handlers/start.py
---------------------
Handles /start command — sends welcome message and shows the bottom keyboard.
"""

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from bot.keyboards.main_menu import get_main_menu
from utils.logger import get_logger

logger = get_logger(__name__)

router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """
    Respond to /start with a welcome message and the bottom navigation keyboard.

    Args:
        message: Incoming /start command message.
    """
    logger.info("Admin %d opened the control panel.", message.from_user.id)
    await message.answer(
        "👋 <b>Hunter Bot — Панель управления</b>\n\n"
        "Используйте кнопки ниже для управления мониторингом.\n\n"
        "📋 <b>Каналы</b> — добавить / включить / отключить / удалить чаты\n"
        "🔌 <b>Распознавание</b> — глобально включить или выключить распознавание\n"
        "📊 <b>Статус</b> — текущее состояние системы",
        reply_markup=get_main_menu(),
    )
