"""
bot/middlewares/auth.py
-----------------------
Outer middleware that blocks every update from non-admin users.

Admin IDs are defined in config.ADMIN_IDS.
Unauthorized requests are silently dropped — no reply is sent.
"""

from typing import Any, Awaitable, Callable, Dict, Optional

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, User

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class AdminOnlyMiddleware(BaseMiddleware):
    """
    Reject any Telegram update whose sender is not in config.ADMIN_IDS.

    Applied as an outer middleware so it runs before all handlers
    and FSM state transitions.
    """

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        """
        Verify admin access before passing the event to the next handler.

        Args:
            handler: Next middleware/handler in the chain.
            event:   Incoming Telegram update object.
            data:    Aiogram context dict (contains 'event_from_user').

        Returns:
            Handler result if authorized, None if not.
        """
        user: Optional[User] = data.get("event_from_user")

        if user is None or user.id not in config.ADMIN_IDS:
            if user:
                logger.warning(
                    "Unauthorized access from user %d (@%s) — blocked.",
                    user.id,
                    user.username or "N/A",
                )
            return None

        return await handler(event, data)
