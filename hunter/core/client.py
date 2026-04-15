"""
core/client.py
--------------
Builds and returns the Telethon TelegramClient instance.
Configured for maximum session longevity:
  - Infinite connection retries with backoff.
  - Realistic device fingerprint to avoid Telegram banning the session.
  - Aggressive flood-sleep handling.
Only one client is created for the entire application lifetime.
"""

from telethon import TelegramClient
from telethon.network import ConnectionTcpFull

import config
from utils.logger import get_logger

logger = get_logger(__name__)


def build_client() -> TelegramClient:
    """
    Create a TelegramClient configured for long-running unattended operation.

    Key settings:
        connection_retries=-1   Retry forever on network drops.
        retry_delay             Seconds to wait between reconnect attempts.
        auto_reconnect=True     Let Telethon handle transient disconnects.
        flood_sleep_threshold   Auto-sleep up to N seconds on FLOOD_WAIT errors.
        device_model / system_version / app_version
                                Mimic a real Telegram Desktop client so Telegram
                                does not flag the session as suspicious.

    Returns:
        TelegramClient: Configured (but not yet connected) client instance.
    """
    logger.debug("Building TelegramClient (session=%s).", config.SESSION_NAME)

    client = TelegramClient(
        config.SESSION_NAME,
        config.API_ID,
        config.API_HASH,
        # Network resilience
        connection=ConnectionTcpFull,
        connection_retries=-1,              # retry forever
        retry_delay=config.RECONNECT_DELAY, # seconds between retries
        auto_reconnect=True,
        # Automatically sleep up to 5 minutes on flood-wait instead of raising
        flood_sleep_threshold=300,
        # Realistic device fingerprint — reduces session termination risk
        device_model="PC 64bit",
        system_version="Windows 10",
        app_version="4.16.4 x64",
        lang_code="en",
        system_lang_code="en-US",
        # Keep receiving updates even while disconnected briefly
        receive_updates=True,
    )
    return client
