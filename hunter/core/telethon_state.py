"""
core/telethon_state.py
----------------------
Singleton module that holds the active TelegramClient reference.

Set once from main.py after authorization.
Read from anywhere that needs the client (e.g. aiogram handlers
that must resolve chat titles via Telethon).
"""

from typing import Optional

from telethon import TelegramClient

# Module-level reference — None until set by main.py
_client: Optional[TelegramClient] = None


def set_client(client: TelegramClient) -> None:
    """
    Store the authorized TelegramClient for global access.

    Must be called from main.py after client.connect() + authorize().

    Args:
        client: Connected and authorized TelegramClient instance.
    """
    global _client  # noqa: PLW0603
    _client = client


def get_client() -> Optional[TelegramClient]:
    """
    Return the active TelegramClient, or None if not yet initialized.

    Returns:
        TelegramClient if set, None otherwise.
    """
    return _client
