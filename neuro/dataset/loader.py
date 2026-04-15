"""
neuro/dataset/loader.py
------------------------
Загрузка и подготовка данных для обучения нейросети.

Умеет загружать:
1. products.json — каталог товаров (основа для синтетических данных).
2. real_messages.txt — реальные сообщения из Telegram-чатов
   (одна строка = одно сообщение).
3. labeled.tsv — размеченные пары (сообщение → правильный товар),
   формат: <сообщение>\\t<правильное_имя_товара>.

Все имена товаров нормализуются через _normalize() из hunter/data/products.py:
убираются эмодзи, флаги, скобки, кириллица переводится в латиницу.
"""

import json
import os
import sys
from typing import Dict, List, Tuple

# Подключаем нормализатор из hunter — убирает флаги, эмодзи, скобки
_hunter_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "hunter"
)
if _hunter_path not in sys.path:
    sys.path.insert(0, os.path.dirname(_hunter_path))

try:
    from hunter.data.products import _normalize as _hunter_normalize
    _NORMALIZE = _hunter_normalize
except ImportError:
    # Fallback если hunter недоступен — базовая очистка
    import re
    _EMOJI_RE = re.compile(
        "[\U0001F1E0-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF]+",
        flags=re.UNICODE,
    )

    def _NORMALIZE(text: str) -> str:
        """Базовая очистка без hunter."""
        text = _EMOJI_RE.sub("", text.lower())
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()


def load_products(products_path: str) -> List[Dict]:
    """
    Загрузить каталог товаров из JSON файла.

    Args:
        products_path: Путь к products.json.

    Returns:
        Список словарей с полями name, price, category, subcategory.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если JSON не является массивом.
    """
    if not os.path.exists(products_path):
        raise FileNotFoundError(
            f"Файл каталога не найден: {products_path}"
        )

    with open(products_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    if not isinstance(products, list):
        raise ValueError("products.json должен содержать JSON-массив.")

    return products


def load_real_messages(messages_path: str) -> List[str]:
    """
    Загрузить реальные сообщения из текстового файла.

    Формат: одно сообщение на строку.
    Пустые строки и строки-комментарии (#) пропускаются.

    Args:
        messages_path: Путь к файлу с сообщениями.

    Returns:
        Список строк (сообщений). Пустой список если файла нет.
    """
    if not os.path.exists(messages_path):
        return []

    messages = []
    with open(messages_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                messages.append(line)

    return messages


def load_labeled_messages(
    labeled_path: str,
) -> List[Tuple[str, str]]:
    """
    Загрузить размеченные сообщения (сообщение → правильный товар).

    Формат файла: TSV (через табуляцию).
    Каждая строка: <сообщение>\\t<правильное_имя_товара>

    Пример:
        айфон 13 128 черный\\tiPhone 13 128 Midnight

    Args:
        labeled_path: Путь к TSV файлу.

    Returns:
        Список кортежей (сообщение, имя_товара).
    """
    if not os.path.exists(labeled_path):
        return []

    pairs = []
    with open(labeled_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0].strip(), parts[1].strip()))

    return pairs


def extract_product_names(products: List[Dict]) -> List[str]:
    """
    Извлечь имена товаров из каталога (lowercase).

    Args:
        products: Список словарей из products.json.

    Returns:
        Список канонических имён товаров.
    """
    names = []
    for product in products:
        name = product.get("name", "").strip()
        if name:
            # Нормализуем через hunter: убираем 🇺🇸, al(s), tl(s/m) и т.д.
            normalized = _NORMALIZE(name)
            if normalized:
                names.append(normalized)
    return names
