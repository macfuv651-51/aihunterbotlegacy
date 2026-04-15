"""
handlers/reply_handler.py
--------------------------
SILENT MODE с подтверждением.

Флоу:
  1. Бот находит товар → отправляет уведомление админу с кнопками:
       [✅ Отправить] [❌ Ошибка]
  2. Если админ нажимает ✅ Отправить → бот шлёт сообщение клиенту.
  3. Если нажимает ❌ Ошибка → сообщение редактируется, добавляется пометка "❌ Ошибка", кнопки убираются.
  4. Пишем запись в data/spy_log.json.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Set

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from telethon import TelegramClient
from telethon.tl.types import User

import config
from bot.instance import bot as tg_bot          # aiogram bot — шлёт уведомления
from data.products import find_all_products
from utils.logger import get_logger
from utils.rate_limiter import RateLimiter

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pending sends storage
# In-memory dict: { "send:{admin_msg_id}:{admin_id}" → { user_id, text } }
# ---------------------------------------------------------------------------
_pending: Dict[str, dict] = {}


def get_pending(key: str) -> dict | None:
    return _pending.get(key)


def pop_pending(key: str) -> dict | None:
    return _pending.pop(key, None)


# ---------------------------------------------------------------------------
# JSON spy-log helper
# ---------------------------------------------------------------------------

def _append_spy_log(entry: dict) -> None:
    """Append one entry to the spy log JSON file (list of objects)."""
    path = config.SPY_LOG_FILE
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        else:
            data = []
        data.append(entry)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.error("spy_log write error: %s", exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def send_reply(
    client: TelegramClient,
    user: User,
    replied_users: Set[int],
    rate_limiter: RateLimiter,
    message_text: str = "",
    message_id: int = 0,
) -> None:
    """
    SILENT MODE с подтверждением.
    Находит товар, уведомляет админа с кнопками [✅ Отправить] [❌ Ошибка].
    """
    if user.bot:
        return
    if not message_text:
        return

    products = find_all_products(message_text)

    # ALWAYS log to spy_log.json
    username   = getattr(user, "username",   None) or ""
    first_name = getattr(user, "first_name", None) or ""
    last_name  = getattr(user, "last_name",  None) or ""
    _append_spy_log({
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "message_id": message_id,
        "user_id":    user.id,
        "username":   username,
        "first_name": first_name,
        "last_name":  last_name,
        "query":      message_text,
        "products":   [p["name"] for p in products] if products else [],
        "matched":    bool(products),
    })

    if not products:
        logger.debug("User %d — no product match for: %s", user.id, message_text[:60])
        return

    lines = []
    for p in products:
        price = str(int(float(p["price"])))
        lines.append(f"{p['name']} — {price}₽")
    reply_text = "\n".join(lines)

    logger.info("[SILENT] Product match for user %d — %d item(s): %s", user.id, len(products), lines)

    notify_text = (
        f"Максим, {message_id}\n\n"
        f"запросил: {message_text}\n"
        f"отправили: {reply_text}"
    )

    for admin_id in config.NOTIFY_ADMIN_IDS:
        try:
            keyboard = InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(text="✅ Отправить", callback_data=f"confirm_send:{user.id}"),
                InlineKeyboardButton(text="❌ Ошибка",    callback_data=f"confirm_err:{user.id}"),
            ]])
            sent = await tg_bot.send_message(admin_id, notify_text, reply_markup=keyboard)
            # Store pending keyed by admin_msg_id + admin_id
            _pending[f"send:{sent.message_id}:{admin_id}"] = {
                "user_id":      user.id,
                "text":         reply_text,
                "rate_limiter": rate_limiter,
            }
            logger.info("Notified admin %d, pending key=send:%d:%d", admin_id, sent.message_id, admin_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to notify admin %d: %s", admin_id, exc)
