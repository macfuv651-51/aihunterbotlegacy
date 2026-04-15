"""
main.py
-------
Entry point. Starts both the Telethon listener and the aiogram
management bot concurrently inside a single asyncio event loop.
"""

import asyncio

import config
from bot.runner import start_bot
from core.client import build_client
from core.session import authorize
from core.telethon_state import set_client
from data.replied_users import load_replied_users
from handlers.message_handler import register_handlers
from utils.logger import get_logger, setup_logger
from utils.rate_limiter import RateLimiter


async def run_telethon() -> None:
    """
    Initialise and run the Telethon userbot with an outer reconnect loop.

    The inner coroutine _telethon_session() handles one connection lifetime.
    If it exits (network error, server restart, etc.) the outer loop waits
    RECONNECT_DELAY seconds and spawns a fresh session automatically.
    MAX_RECONNECT_ATTEMPTS=-1 means retry forever.
    """
    logger = get_logger(__name__)

    # Load persisted state once — shared across all reconnect iterations
    replied_users = load_replied_users()
    logger.info("Loaded %d previously replied users.", len(replied_users))
    rate_limiter = RateLimiter()

    attempt = 0

    while True:
        attempt += 1
        logger.info("Telethon session attempt #%d ...", attempt)

        try:
            await _telethon_session(replied_users, rate_limiter)
        except Exception as exc:  # noqa: BLE001
            logger.error("Telethon session crashed: %s", exc, exc_info=True)

        # Check whether we've exhausted the retry budget
        if (
            config.MAX_RECONNECT_ATTEMPTS != -1
            and attempt >= config.MAX_RECONNECT_ATTEMPTS
        ):
            logger.critical("Max reconnect attempts reached — giving up.")
            break

        logger.info(
            "Reconnecting in %d seconds ...", config.RECONNECT_DELAY
        )
        await asyncio.sleep(config.RECONNECT_DELAY)


async def _telethon_session(
    replied_users: set,
    rate_limiter: RateLimiter,
) -> None:
    """
    Run a single Telethon connection lifetime.

    Builds the client, authorizes, registers handlers, then blocks
    until Telegram disconnects us for any reason.

    Args:
        replied_users: Shared in-memory set passed from the outer loop.
        rate_limiter:  Shared RateLimiter instance.
    """
    logger = get_logger(__name__)

    client = build_client()

    # Register handlers BEFORE connecting so that no incoming updates are missed.
    # The receive loop starts inside client.connect(), so any message that
    # arrives during the auth / GetDifference phase is caught from the start.
    register_handlers(client, replied_users, rate_limiter)

    await client.connect()
    await authorize(client)

    # Share the authorized client so aiogram handlers can resolve chat titles
    set_client(client)

    logger.info("Telethon listener running ...")
    await client.run_until_disconnected()

    # Graceful cleanup
    await client.disconnect()
    logger.info("Telethon client disconnected cleanly.")


async def main() -> None:
    """
    Wire up logging and launch both services concurrently.

    asyncio.gather runs run_telethon() and start_bot() in the same
    event loop. Either task crashing will propagate and stop both.
    """
    setup_logger()
    logger = get_logger(__name__)
    logger.info("Starting hunter bot (Telethon + aiogram) ...")

    await asyncio.gather(
        run_telethon(),
        start_bot(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Manual stop (Ctrl+C / terminal interrupt): exit quietly.
        get_logger(__name__).info("Shutdown requested by user.")
