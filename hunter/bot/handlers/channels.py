"""
bot/handlers/channels.py
------------------------
Manages the list of monitored Telegram chats.

Features:
  - Show all chats with enable/disable toggles and delete buttons.
  - Chat title is resolved via Telethon when a chat is added.
  - Add a new chat via FSM (multi-step conversation).
  - Toggle individual chats on/off.
  - Delete a chat from the list.
"""

from aiogram import F, Router
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from bot.keyboards.main_menu import BTN_CHANNELS, get_main_menu
from core.telethon_state import get_client
from data.app_state import add_chat, get_chats, remove_chat, toggle_chat
from utils.logger import get_logger

logger = get_logger(__name__)
router = Router()


class AddChatStates(StatesGroup):
    """FSM state group for the 'add chat' conversation flow."""

    waiting_for_chat_id = State()


class ChatCb(CallbackData, prefix="ch"):
    """
    Callback data for inline chat management buttons.

    Attributes:
        action:  'toggle' or 'delete'.
        chat_id: The stored chat identifier string.
    """

    action: str
    chat_id: str


async def _resolve_title(chat_id: str) -> str:
    """
    Attempt to resolve the human-readable title of a chat via Telethon.

    Strategy:
        1. Try get_entity() — works for public chats and already-cached entities.
        2. If that fails (private group not in cache), scan iter_dialogs() to find
           the chat by matching its numeric ID or username.
        3. Fall back to the raw chat_id string if nothing works.

    Args:
        chat_id: Username (e.g. '@my_group') or numeric ID string.

    Returns:
        str: Chat title, or chat_id if resolution fails.
    """
    client = get_client()
    if client is None:
        return chat_id

    is_numeric = chat_id.lstrip("-").isdigit()
    entity_input = int(chat_id) if is_numeric else chat_id

    # --- attempt 1: direct entity lookup (fast, works for public/cached) ---
    try:
        entity = await client.get_entity(entity_input)
        title = getattr(entity, "title", None) or getattr(entity, "username", None)
        if title:
            return title
    except Exception:  # noqa: BLE001
        pass  # fall through to dialog scan

    # --- attempt 2: scan dialogs (finds private groups by numeric ID) ---
    logger.debug(
        "Direct entity lookup failed for '%s' — scanning dialogs ...", chat_id
    )
    try:
        async for dialog in client.iter_dialogs():
            d = dialog.entity
            d_id = getattr(d, "id", None)
            d_username = getattr(d, "username", None) or ""

            matched = (
                (is_numeric and d_id is not None and str(d_id) in chat_id)
                or (not is_numeric and d_username.lower() == chat_id.lstrip("@").lower())
            )

            if matched:
                return getattr(d, "title", None) or d_username or chat_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("Dialog scan failed for '%s': %s", chat_id, exc)

    # --- fallback ---
    logger.warning("Could not resolve title for '%s' — using raw ID.", chat_id)
    return chat_id


def _channels_keyboard() -> InlineKeyboardMarkup:
    """
    Build an inline keyboard listing all monitored chats.

    Each chat gets:
      - A toggle button showing ✅/❌ and the chat TITLE (not raw ID).
      - A 🗑 delete button.
    A final row contains '➕ Добавить чат'.

    Returns:
        InlineKeyboardMarkup with one row per chat plus an add-button row.
    """
    chats = get_chats()
    rows = []

    for chat in chats:
        status = "✅" if chat.get("enabled", True) else "❌"
        # Use stored title; fall back to id for legacy entries without title
        display = chat.get("title") or chat["id"]
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{status} {display}",
                    callback_data=ChatCb(action="toggle", chat_id=chat["id"]).pack(),
                ),
                InlineKeyboardButton(
                    text="🗑",
                    callback_data=ChatCb(action="delete", chat_id=chat["id"]).pack(),
                ),
            ]
        )

    rows.append(
        [InlineKeyboardButton(text="➕ Добавить чат", callback_data="ch_add")]
    )

    return InlineKeyboardMarkup(inline_keyboard=rows)


@router.message(F.text == BTN_CHANNELS)
async def show_channels(message: Message, state: FSMContext) -> None:
    """
    Display the current monitored chat list with management controls.

    Clears any active FSM state before showing the list.

    Args:
        message: Incoming message with text '📋 Channels'.
        state:   FSM context (cleared on entry).
    """
    await state.clear()

    chats = get_chats()
    header = (
        f"<b>📋 Отслеживаемые чаты</b> — всего {len(chats)}\n\n"
        "✅ / ❌ включить-выключить  |  🗑 удалить"
        if chats
        else "<b>📋 Отслеживаемые чаты</b>\n\nЧатов пока нет — нажмите <b>➕ Добавить чат</b>."
    )

    await message.answer(header, reply_markup=_channels_keyboard())


@router.callback_query(F.data == "ch_add")
async def prompt_add_chat(callback: CallbackQuery, state: FSMContext) -> None:
    """
    Ask the admin to enter a chat username or numeric ID.

    Sets FSM state to AddChatStates.waiting_for_chat_id.

    Args:
        callback: Callback query from the '➕ Add chat' inline button.
        state:    FSM context transitioned to waiting_for_chat_id.
    """
    await state.set_state(AddChatStates.waiting_for_chat_id)
    await callback.message.answer(
        "Введите <b>username</b> чата или <b>числовой ID</b>:\n\n"
        "Примеры:\n"
        "  <code>@my_group</code>\n"
        "  <code>-1001234567890</code>"
    )
    await callback.answer()


@router.message(AddChatStates.waiting_for_chat_id)
async def receive_chat_id(message: Message, state: FSMContext) -> None:
    """
    Process the chat identifier provided by the admin and add it to state.

    1. Normalise the identifier.
    2. Resolve the chat title via Telethon.
    3. Persist to app_state with the resolved title.

    Args:
        message: Admin's reply containing the chat username or numeric ID.
        state:   FSM context (cleared after processing).
    """
    raw: str = (message.text or "").strip()

    if not raw:
        await message.answer("⚠️ Отправьте корректный username или числовой ID.")
        return

    # Normalise identifier
    if raw.lstrip("-").isdigit():
        chat_id = raw
    elif not raw.startswith("@"):
        chat_id = f"@{raw}"
    else:
        chat_id = raw

    # Resolve human-readable title before saving
    title = await _resolve_title(chat_id)

    added = add_chat(chat_id, title=title)
    await state.clear()

    if added:
        logger.info("Admin added chat '%s' (title: %s).", chat_id, title)
        await message.answer(
            f"✅ <b>{title}</b> (<code>{chat_id}</code>) добавлен и включен.",
            reply_markup=get_main_menu(),
        )
    else:
        await message.answer(
            f"⚠️ Чат <code>{chat_id}</code> уже есть в списке.",
            reply_markup=get_main_menu(),
        )

    await message.answer(
        "<b>📋 Обновленный список чатов:</b>",
        reply_markup=_channels_keyboard(),
    )


@router.callback_query(ChatCb.filter(F.action == "toggle"))
async def toggle_channel(
    callback: CallbackQuery, callback_data: ChatCb
) -> None:
    """
    Toggle the enabled/disabled state of a monitored chat.

    Updates the inline keyboard in-place to reflect the new state.

    Args:
        callback:      Callback query from a toggle button.
        callback_data: Parsed ChatCb with action='toggle' and chat_id.
    """
    new_state = toggle_chat(callback_data.chat_id)

    if new_state is None:
        await callback.answer("⚠️ Чат не найден.", show_alert=True)
        return

    status = "включен ✅" if new_state else "выключен ❌"
    logger.info("Admin toggled chat '%s' → %s.", callback_data.chat_id, status)
    await callback.answer(f"Чат теперь {status}.")

    await callback.message.edit_reply_markup(reply_markup=_channels_keyboard())


@router.callback_query(ChatCb.filter(F.action == "delete"))
async def delete_channel(
    callback: CallbackQuery, callback_data: ChatCb
) -> None:
    """
    Remove a chat from the monitored list.

    Updates the inline keyboard in-place after deletion.

    Args:
        callback:      Callback query from a delete button.
        callback_data: Parsed ChatCb with action='delete' and chat_id.
    """
    removed = remove_chat(callback_data.chat_id)

    if not removed:
        await callback.answer("⚠️ Чат не найден.", show_alert=True)
        return

    logger.info("Admin deleted chat '%s'.", callback_data.chat_id)
    await callback.answer("🗑 Чат удален.")

    await callback.message.edit_reply_markup(reply_markup=_channels_keyboard())
