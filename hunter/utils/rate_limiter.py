"""
utils/rate_limiter.py
---------------------
Token-bucket rate limiter to avoid Telegram flood bans.
Tracks per-hour message count and enforces minimum delay between sends.
"""

import asyncio
import time
from collections import deque
from typing import Deque

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Enforces two limits simultaneously:
      1. Minimum pause between consecutive messages (DELAY_BETWEEN_MESSAGES).
      2. Maximum number of messages per rolling 1-hour window (MAX_MESSAGES_PER_HOUR).
    """

    def __init__(self) -> None:
        """
        Initialize the rate limiter with values from config.
        """
        self._min_delay: int = config.DELAY_BETWEEN_MESSAGES
        self._max_per_hour: int = config.MAX_MESSAGES_PER_HOUR
        self._last_sent: float = 0.0
        # Timestamps of messages sent within the last hour
        self._sent_times: Deque[float] = deque()

    def _purge_old_timestamps(self) -> None:
        """
        Remove timestamps older than 1 hour from the sliding window.
        """
        one_hour_ago = time.time() - 3600
        while self._sent_times and self._sent_times[0] < one_hour_ago:
            self._sent_times.popleft()

    def is_allowed(self) -> bool:
        """
        Check whether sending a message right now is within both limits.

        Returns:
            bool: True if sending is allowed.
        """
        self._purge_old_timestamps()
        time_since_last = time.time() - self._last_sent
        if time_since_last < self._min_delay:
            return False
        if len(self._sent_times) >= self._max_per_hour:
            return False
        return True

    async def wait_until_allowed(self) -> None:
        """
        Async-sleep until sending is permitted by both limits.
        Uses asyncio.sleep so the event loop is never blocked.
        Logs a warning each time it has to wait.
        """
        while not self.is_allowed():
            self._purge_old_timestamps()
            time_since_last = time.time() - self._last_sent
            delay_remaining = self._min_delay - time_since_last

            if len(self._sent_times) >= self._max_per_hour:
                sleep_for = max(
                    self._sent_times[0] + 3600 - time.time(), 1
                )
                logger.warning(
                    "Hourly limit reached (%d msgs). Sleeping %.0fs.",
                    self._max_per_hour,
                    sleep_for,
                )
            else:
                sleep_for = max(delay_remaining, 1)
                logger.warning(
                    "Min delay not elapsed. Sleeping %.0fs.", sleep_for
                )

            await asyncio.sleep(sleep_for)

    def record_send(self) -> None:
        """
        Record a successful message send.
        Must be called right after every successful send.
        """
        now = time.time()
        self._last_sent = now
        self._sent_times.append(now)



