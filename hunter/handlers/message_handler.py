"""
handlers/message_handler.py
-----------------------------
Listens for ALL incoming messages and decides whether to trigger
a reply based on:
  1. Global Telethon recognition flag (app_state.is_telethon_enabled).
  2. Whether the source chat is in the monitored list and enabled.
  3. Keyword matching against data/keywords.json groups.
"""

from typing import Set

from telethon import TelegramClient, events
from telethon.tl.types import User

from data.app_state import is_chat_monitored, is_telethon_enabled
from handlers.reply_handler import send_reply
from utils.keywords import find_matches
from utils.logger import get_logger
from utils.rate_limiter import RateLimiter

logger = get_logger(__name__)


def register_handlers(
    client: TelegramClient,
    replied_users: Set[int],
    rate_limiter: RateLimiter,
) -> None:
    """
    Attach all Telethon event handlers to the client.

    Registers a catch-all NewMessage handler (no chats filter) so that
    newly added chats are picked up without restarting the bot.
    Dynamic filtering is done inside the handler using app_state.

    Args:
        client:         Active Telethon client.
        replied_users:  Shared in-memory set of already-replied user IDs.
        rate_limiter:   Shared RateLimiter instance.
    """

    @client.on(events.NewMessage)
    async def on_new_message(event: events.NewMessage.Event) -> None:
        """
        Handle every incoming message across all chats.

        1. Skip if Telethon recognition is globally disabled.
        2. Skip if the message is from a chat not in the monitored list
           or the chat is currently disabled.
        3. Skip empty messages.
        4. Run keyword matching — skip if no match.
        5. Resolve the sender; skip bots and anonymous posts.
        6. Log all matched groups/keywords.
        7. Delegate to send_reply.
        """
        # Global kill-switch — checked on every message with zero overhead
        if not is_telethon_enabled():
            return

        # Resolve the chat this message came from
        chat = await event.get_chat()
        chat_numeric_id: int = event.chat_id or 0
        chat_username: str = getattr(chat, "username", None) or ""

        if not is_chat_monitored(chat_numeric_id, chat_username):
            logger.debug(
                "Message from unmonitored chat id=%s username='%s' — skipping.",
                chat_numeric_id,
                chat_username,
            )
            return

        message_text: str = event.raw_text or ""

        if not message_text.strip():
            return

        matches = find_matches(message_text)

        if not matches:
            return

        # Resolve sender to a full User object
        sender = await event.get_sender()

        if not isinstance(sender, User):
            # Channel re-post or anonymous admin — skip
            return

        match_summary = ", ".join(
            f"[{m.group}] '{m.keyword}'" for m in matches
        )
        logger.info(
            "User %d (@%s) — %d match(es): %s | chat: %s | text: %.300s",
            sender.id,
            getattr(sender, "username", "N/A"),
            len(matches),
            match_summary,
            chat_username or chat_numeric_id,
            message_text,
        )

        await send_reply(client, sender, replied_users, rate_limiter, message_text,
                          message_id=event.message.id)

    logger.info("Catch-all message handler registered (dynamic chat filtering).")
