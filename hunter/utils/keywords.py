"""
utils/keywords.py
-----------------
Loads keyword groups from data/keywords.json and provides
matching utilities that return detailed match information.

JSON structure expected:
    {
        "group_name": ["keyword1", "keyword2", ...],
        ...
    }
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List

import config
from utils.logger import get_logger

logger = get_logger(__name__)

# In-memory cache: loaded once on first use, reloaded if the file changes.
_keywords_cache: Dict[str, List[str]] = {}
_cache_mtime: float = 0.0


@dataclass
class MatchResult:
    """
    Represents a single keyword match inside a message.

    Attributes:
        group:   Name of the keyword group (e.g. "buy_intent").
        keyword: The exact keyword string that was found.
    """

    group: str
    keyword: str


def _load_from_disk() -> Dict[str, List[str]]:
    """
    Read and parse the keywords JSON file from disk.

    Returns:
        Dict mapping group name to list of keyword strings.

    Raises:
        FileNotFoundError: If the keywords file does not exist.
        ValueError:        If the JSON structure is invalid.
    """
    path = config.KEYWORDS_FILE

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Keywords file not found: {path}. "
            "Create it or update KEYWORDS_FILE in config.py."
        )

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError("keywords.json must be a JSON object (dict at top level).")

    # Normalise: strip whitespace, lowercase every keyword
    normalised: Dict[str, List[str]] = {}
    for group, words in data.items():
        if not isinstance(words, list):
            logger.warning("Group '%s' is not a list — skipping.", group)
            continue
        normalised[group] = [str(w).strip().lower() for w in words if str(w).strip()]

    total = sum(len(v) for v in normalised.values())
    logger.info(
        "Loaded %d keyword groups (%d keywords total) from %s.",
        len(normalised),
        total,
        path,
    )
    return normalised


def get_keywords() -> Dict[str, List[str]]:
    """
    Return the keyword dictionary, reloading from disk if the file changed.

    Returns:
        Dict mapping group name to list of lowercase keyword strings.
    """
    global _keywords_cache, _cache_mtime  # noqa: PLW0603

    path = config.KEYWORDS_FILE
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0

    if not _keywords_cache or mtime != _cache_mtime:
        _keywords_cache = _load_from_disk()
        _cache_mtime = mtime

    return _keywords_cache


def find_matches(text: str) -> List[MatchResult]:
    """
    Find all keyword matches in the given text across every group.

    Scans each group independently so multiple matches from
    different groups are all reported.

    Args:
        text: Raw message text from Telegram.

    Returns:
        List of MatchResult objects (one per matched keyword).
        Empty list if no keywords match.
    """
    lowered = text.lower()
    matches: List[MatchResult] = []

    for group, keywords in get_keywords().items():
        for keyword in keywords:
            if keyword in lowered:
                matches.append(MatchResult(group=group, keyword=keyword))

    return matches


def contains_keyword(text: str) -> bool:
    """
    Quick boolean check: does the text match any keyword in any group?

    Args:
        text: Raw message text from Telegram.

    Returns:
        bool: True if at least one keyword is found.
    """
    return len(find_matches(text)) > 0
