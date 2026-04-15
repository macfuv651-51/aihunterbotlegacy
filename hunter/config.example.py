"""
hunter/config.example.py
-------------------------
ШАБЛОН конфигурации. Скопируй в config.py и заполни своими данными:
    cp config.example.py config.py

Все значения с placeholder'ами нужно заменить реальными.
"""

# Telegram API credentials
# Получить на: https://my.telegram.org/apps
API_ID: int = 0                  # Твой api_id (целое число)
API_HASH: str = "YOUR_API_HASH"  # Твой api_hash (строка)

# Номер телефона аккаунта который запускает бота
PHONE: str = "+79000000000"

# Session
SESSION_NAME: str = "seller_session"

# Reconnection
RECONNECT_DELAY: int = 5
MAX_RECONNECT_ATTEMPTS: int = -1

# Rate limiting
DELAY_BETWEEN_MESSAGES: int = 5
MAX_MESSAGES_PER_HOUR: int = 20

# Management bot — токен от @BotFather
BOT_TOKEN: str = "YOUR_BOT_TOKEN"

# Telegram user IDs допущенных к управлению ботом
ADMIN_IDS: list = [123456789]

# Файлы данных
import os
_BASE = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_FILE:        str = os.path.join(_BASE, "data", "products.json")
KEYWORDS_FILE:        str = os.path.join(_BASE, "data", "keywords.json")
REPLIED_USERS_FILE:   str = os.path.join(_BASE, "data", "replied_users.json")
SPY_LOG_FILE:         str = os.path.join(_BASE, "data", "spy_log.json")
APP_STATE_FILE:       str = os.path.join(_BASE, "data", "app_state.json")
ERROR_SPY_LOG_FILE:   str = os.path.join(_BASE, "data", "error_spy_log.txt")

# Logger
LOG_DIR: str = os.path.join(_BASE, "logs")
LOG_FILE: str = os.path.join(LOG_DIR, "seller.log")
LOG_MAX_BYTES: int = 5 * 1024 * 1024
LOG_BACKUP_COUNT: int = 3
