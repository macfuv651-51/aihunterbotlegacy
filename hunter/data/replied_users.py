"""
data/replied_users.py
---------------------
Persistent storage for user IDs that have already been messaged.
Backed by a JSON file defined in config.REPLIED_USERS_FILE.
"""

import json
import os
from typing import Set

import config
from utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_file() -> None:
    """
    Create the storage file and its parent directory if they do not exist.
    """
    os.makedirs(os.path.dirname(config.REPLIED_USERS_FILE), exist_ok=True)
    if not os.path.exists(config.REPLIED_USERS_FILE):
        with open(config.REPLIED_USERS_FILE, "w", encoding="utf-8") as fh:
            json.dump([], fh)


def load_replied_users() -> Set[int]:
    """
    Load the set of already-messaged user IDs from disk.

    Returns:
        Set[int]: IDs of users who have already received a reply.
    """
    _ensure_file()
    with open(config.REPLIED_USERS_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return set(data)


def save_replied_users(users: Set[int]) -> None:
    """
    Persist the set of replied user IDs to disk.

    Args:
        users: Updated set of user IDs to save.
    """
    _ensure_file()
    with open(config.REPLIED_USERS_FILE, "w", encoding="utf-8") as fh:
        json.dump(list(users), fh, indent=2)
    logger.debug("Saved %d replied users to disk.", len(users))


def add_replied_user(user_id: int, users: Set[int]) -> None:
    """
    Add a single user ID to the set and immediately persist it.

    Args:
        user_id: Telegram user ID to add.
        users:   The in-memory set to update (mutated in place).
    """
    users.add(user_id)
    save_replied_users(users)
    logger.debug("Marked user %d as replied.", user_id)
