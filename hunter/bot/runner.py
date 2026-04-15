"""
bot/runner.py
-------------
Initialises the aiogram bot and starts the polling loop.

Called from main.py and run as an asyncio task alongside Telethon.
"""

import asyncio

import config
from bot.handlers import register_all_routers
from bot.instance import bot, dp
from bot.middlewares.auth import AdminOnlyMiddleware
from utils.logger import get_logger

logger = get_logger(__name__)


async def _bot_session() -> None:
    """Run one polling session. Blocks until the session ends."""
    dp.update.outer_middleware(AdminOnlyMiddleware())
    register_all_routers(dp)
    logger.info("Starting aiogram management bot ...")
    await dp.start_polling(bot, handle_signals=False)


async def start_bot() -> None:
    """
    Wire up middlewares and routers, then begin polling Telegram for updates.

    Wraps the polling loop with the same infinite-retry logic used by
    run_telethon(): on any network error the bot waits RECONNECT_DELAY
    seconds and reconnects automatically instead of crashing the process.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            await _bot_session()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "aiogram bot session crashed (attempt %d): %s", attempt, exc
            )

        if (
            config.MAX_RECONNECT_ATTEMPTS != -1
            and attempt >= config.MAX_RECONNECT_ATTEMPTS
        ):
            logger.critical("aiogram: max reconnect attempts reached — giving up.")
            break

        logger.info(
            "aiogram: reconnecting in %d seconds ...", config.RECONNECT_DELAY
        )
        await asyncio.sleep(config.RECONNECT_DELAY)
