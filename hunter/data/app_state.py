"""
data/app_state.py
-----------------
Shared application state persisted to JSON.

Both the Telethon listener and the aiogram management bot read/write this file.
A threading.Lock protects all disk writes so concurrent asyncio tasks are safe.

State schema:
    {
        "telethon_enabled": true,
        "chats": [
            {"id": "@some_group", "title": "My Group", "enabled": true},
            ...
        ]
    }
"""

import json
import os
import threading
from typing import Any, Dict, List, Optional

import config
from utils.logger import get_logger

logger = get_logger(__name__)

# Write lock — asyncio tasks run in one thread but we protect saves anyway
_lock = threading.Lock()

_DEFAULT_STATE: Dict[str, Any] = {
    "telethon_enabled": True,
    "chats": [],
}


def _ensure_file() -> None:
    """
    Create the state file and its parent directory if they do not exist.
    Initialises the file with default values.
    """
    dirpath = os.path.dirname(config.APP_STATE_FILE)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    if not os.path.exists(config.APP_STATE_FILE):
        with open(config.APP_STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump(_DEFAULT_STATE, fh, indent=2, ensure_ascii=False)


def _read() -> Dict[str, Any]:
    """
    Read and return the raw state dict from disk.

    Returns:
        Dict with keys 'telethon_enabled' and 'chats'.
    """
    _ensure_file()
    with open(config.APP_STATE_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write(state: Dict[str, Any]) -> None:
    """
    Persist the state dict to disk (thread-safe).

    Args:
        state: Full state dict to save.
    """
    with _lock:
        with open(config.APP_STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, ensure_ascii=False)
    logger.debug("App state saved.")


def is_telethon_enabled() -> bool:
    """
    Return whether Telethon message recognition is globally active.

    Returns:
        bool: True if enabled.
    """
    return _read().get("telethon_enabled", True)


def set_telethon_enabled(enabled: bool) -> None:
    """
    Enable or disable Telethon message recognition globally.

    Args:
        enabled: True to enable, False to disable.
    """
    state = _read()
    state["telethon_enabled"] = enabled
    _write(state)
    logger.info("Telethon recognition → %s.", "ON" if enabled else "OFF")


def get_chats() -> List[Dict[str, Any]]:
    """
    Return the full list of monitored chat entries.

    Returns:
        List of dicts: [{"id": str, "enabled": bool}, ...]
    """
    return _read().get("chats", [])


def add_chat(chat_id: str, title: Optional[str] = None) -> bool:
    """
    Add a new chat to the monitored list (enabled by default).

    Args:
        chat_id: Username (e.g. '@my_group') or numeric ID string.
        title:   Human-readable chat name shown in the management bot.
                 Falls back to chat_id if not provided.

    Returns:
        bool: True if added, False if the chat was already present.
    """
    state = _read()
    chats: List[Dict[str, Any]] = state.setdefault("chats", [])

    if any(c["id"] == chat_id for c in chats):
        return False

    chats.append({"id": chat_id, "title": title or chat_id, "enabled": True})
    _write(state)
    logger.info("Chat '%s' (%s) added to monitored list.", chat_id, title or chat_id)
    return True


def remove_chat(chat_id: str) -> bool:
    """
    Remove a chat from the monitored list.

    Args:
        chat_id: The chat identifier to remove.

    Returns:
        bool: True if removed, False if not found.
    """
    state = _read()
    before = len(state.get("chats", []))
    state["chats"] = [c for c in state.get("chats", []) if c["id"] != chat_id]

    if len(state["chats"]) == before:
        return False

    _write(state)
    logger.info("Chat '%s' removed from monitored list.", chat_id)
    return True


def toggle_chat(chat_id: str) -> Optional[bool]:
    """
    Toggle the enabled/disabled state of a monitored chat.

    Args:
        chat_id: The chat identifier to toggle.

    Returns:
        bool: New enabled state after toggle, or None if chat not found.
    """
    state = _read()
    for chat in state.get("chats", []):
        if chat["id"] == chat_id:
            chat["enabled"] = not chat["enabled"]
            _write(state)
            logger.info("Chat '%s' → %s.", chat_id, "ON" if chat["enabled"] else "OFF")
            return chat["enabled"]
    return None


def _normalize_id(raw_id: str) -> str:
    """
    Strip the Telegram Bot API '-100' supergroup prefix if present.

    Telethon reports supergroup IDs as e.g. -5181704594,
    while the Bot API (and user input) uses -1005181704594.
    Normalizing both sides to the bare numeric ID makes matching reliable.

    Args:
        raw_id: String representation of a chat ID.

    Returns:
        str: ID without the '-100' prefix, e.g. '-5181704594'.
    """
    if raw_id.startswith("-100"):
        return "-" + raw_id[4:]
    return raw_id


def is_chat_monitored(chat_id: int, username: Optional[str] = None) -> bool:
    """
    Check whether a given chat is monitored and currently enabled.

    Matches by numeric ID (normalized — handles -100 prefix mismatch)
    or by @username (case-insensitive).

    Args:
        chat_id:  Numeric Telegram chat ID as reported by Telethon.
        username: Optional username of the chat (with or without '@').

    Returns:
        bool: True if the chat is actively monitored.
    """
    chats = get_chats()

    # Normalize the incoming ID so both '-5181704594' and '-1005181704594' match
    normalized_incoming = _normalize_id(str(chat_id))

    if username and not username.startswith("@"):
        username = f"@{username}"

    for chat in chats:
        if not chat.get("enabled", True):
            continue
        stored = chat["id"]
        if _normalize_id(stored) == normalized_incoming:
            return True
        if username and stored.lower() == username.lower():
            return True

    return False
    return False
