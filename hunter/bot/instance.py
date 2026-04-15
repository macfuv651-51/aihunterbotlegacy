"""
bot/instance.py
---------------
Creates the single Bot and Dispatcher instances used across all handlers.

Import 'bot' and 'dp' from here — never create new instances elsewhere.
"""

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

import config

# Single Bot instance for the whole application
# timeout=60 prevents TelegramNetworkError on slow connections
bot = Bot(
    token=config.BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    session=AiohttpSession(timeout=60),
)

# Single Dispatcher — in-memory FSM storage (no Redis needed)
dp = Dispatcher(storage=MemoryStorage())
