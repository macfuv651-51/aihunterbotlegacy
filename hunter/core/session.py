"""
core/session.py
---------------
Handles Telegram authorization flow.
Prompts for a phone code (and 2FA password if needed) on first launch.
"""

from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

import config
from utils.logger import get_logger

logger = get_logger(__name__)


async def authorize(client: TelegramClient) -> None:
    """
    Ensure the client is fully authorized.

    Sends a sign-in code to the phone number from config on first run.
    If two-factor authentication is enabled, prompts for the password.

    Args:
        client: Connected but possibly unauthorized TelegramClient.
    """
    if await client.is_user_authorized():
        logger.info("Session already authorized.")
        return

    logger.info("Sending sign-in code to %s ...", config.PHONE)
    await client.send_code_request(config.PHONE)

    code = input("Enter the code you received in Telegram: ").strip()

    try:
        await client.sign_in(config.PHONE, code)
    except SessionPasswordNeededError:
        # Account has two-factor authentication enabled
        password = input("Enter your 2FA password: ").strip()
        await client.sign_in(password=password)

    logger.info("Authorization successful.")
