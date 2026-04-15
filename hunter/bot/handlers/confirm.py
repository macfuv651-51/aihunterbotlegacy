"""
bot/handlers/confirm.py
-----------------------
Handles admin confirmation buttons for pending product replies.

Callback data formats:
  confirm_send:{user_id}  — admin approved, send reply to user
  confirm_err:{user_id}   — admin marked as error, edit message and drop pending
"""

from aiogram import F, Router
from aiogram.types import CallbackQuery

from core.telethon_state import get_client
from handlers.reply_handler import pop_pending
from utils.logger import get_logger

logger = get_logger(__name__)
router = Router()


@router.callback_query(F.data.startswith("confirm_send:"))
async def on_confirm_send(callback: CallbackQuery) -> None:
    """Admin approved — send the reply to the user via Telethon."""
    admin_id = callback.from_user.id
    msg_id   = callback.message.message_id
    key      = f"send:{msg_id}:{admin_id}"

    pending = pop_pending(key)
    if not pending:
        await callback.answer("⚠️ Уже отправлено или устарело.", show_alert=False)
        # Remove buttons anyway
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        return

    user_id      = pending["user_id"]
    text         = pending["text"]
    rate_limiter = pending["rate_limiter"]

    client = get_client()
    if not client:
        await callback.answer("❌ Telethon не подключён.", show_alert=True)
        return

    try:
        await rate_limiter.wait_until_allowed()
        await client.send_message(user_id, text)
        rate_limiter.record_send()
        logger.info("Sent reply to user %d after admin confirm.", user_id)

        # Edit admin message — remove buttons, add ✅
        original = callback.message.text or ""
        await callback.message.edit_text(
            original + "\n\n✅ Отправлено",
            reply_markup=None,
        )
        await callback.answer("✅ Отправлено!")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to send reply to user %d: %s", user_id, exc)
        await callback.answer("❌ Ошибка отправки.", show_alert=True)


@router.callback_query(F.data.startswith("confirm_err:"))
async def on_confirm_err(callback: CallbackQuery) -> None:
    """Admin marked as error — edit message, drop pending."""
    admin_id = callback.from_user.id
    msg_id   = callback.message.message_id
    key      = f"send:{msg_id}:{admin_id}"

    pop_pending(key)  # discard pending silently

    try:
        original = callback.message.text or ""
        await callback.message.edit_text(
            original + "\n\n❌ Ошибка",
            reply_markup=None,
        )
    except Exception:
        pass

    await callback.answer("Отмечено как ошибка.")
    logger.info("Admin %d marked message %d as error.", admin_id, msg_id)
