"""
neuro/augment/noise.py
----------------------
Функции зашумления текста для генерации обучающих данных.

Каждая функция имитирует реальные человеческие ошибки:
- Опечатки (замена на соседнюю клавишу)
- Пропуск символов (промахнулся мимо клавиши)
- Дублирование символов (залипание клавиши)
- Перестановка соседних букв (быстрый набор)

Используется в generator.py для создания синтетических
вариантов товаров.
"""

import random
import string
from typing import List


# ─── Раскладка клавиатуры ─────────────────────────────────────────────────────
# Символ → соседние клавиши (QWERTY-раскладка)

_KEYBOARD_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrds", "r": "etdf",
    "t": "ryfg", "y": "tugh", "u": "yijh", "i": "uojk",
    "o": "iplk", "p": "ol",
    "a": "qwsz", "s": "weadzx", "d": "ersfxc", "f": "rtdgcv",
    "g": "tyfhvb", "h": "uygjbn", "j": "iuhknm", "k": "iojlm",
    "l": "opk",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb",
    "b": "vghn", "n": "bhjm", "m": "njk",
    "1": "2q", "2": "13w", "3": "24e", "4": "35r", "5": "46t",
    "6": "57y", "7": "68u", "8": "79i", "9": "80o", "0": "9p",
}

# Кириллическая раскладка (ЙЦУКЕН)
_KEYBOARD_NEIGHBORS_RU = {
    "й": "цф", "ц": "йуыв", "у": "цкыва", "к": "уеап",
    "е": "кнпр", "н": "егро", "г": "нштол", "ш": "гщдлб",
    "щ": "шзжд", "з": "щхж", "х": "зъ",
    "ф": "йцыя", "ы": "цувач", "в": "укапс", "а": "кепрм",
    "п": "енрои", "р": "нготи", "о": "гшлдт", "л": "шщжб",
    "д": "щзжэю", "ж": "зхэ",
    "я": "фыч", "ч": "ыясв", "с": "вамч", "м": "апит",
    "и": "пртоь", "т": "роль", "ь": "олдб", "б": "лджю",
    "ю": "джэ",
}


def swap_adjacent_chars(text: str) -> str:
    """
    Поменять местами два соседних символа.

    Имитирует опечатку при быстром наборе текста.
    Пример: 'iphone' → 'ihpone'

    Args:
        text: Исходная строка.

    Returns:
        Строка с переставленными символами.
    """
    if len(text) < 2:
        return text
    chars = list(text)
    idx = random.randint(0, len(chars) - 2)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def delete_random_char(text: str) -> str:
    """
    Удалить случайный символ из строки.

    Имитирует пропуск клавиши при наборе.
    Пример: 'iphone' → 'iphne'

    Args:
        text: Исходная строка.

    Returns:
        Строка с удалённым символом.
    """
    if len(text) < 3:
        return text
    idx = random.randint(0, len(text) - 1)
    return text[:idx] + text[idx + 1:]


def duplicate_random_char(text: str) -> str:
    """
    Продублировать случайный символ.

    Имитирует залипание клавиши.
    Пример: 'iphone' → 'iphhone'

    Args:
        text: Исходная строка.

    Returns:
        Строка с продублированным символом.
    """
    if not text:
        return text
    idx = random.randint(0, len(text) - 1)
    return text[:idx] + text[idx] + text[idx:]


def insert_random_char(text: str) -> str:
    """
    Вставить случайный символ в случайную позицию.

    Имитирует случайное нажатие лишней клавиши.
    Пример: 'iphone' → 'iphoqne'

    Args:
        text: Исходная строка.

    Returns:
        Строка со вставленным случайным символом.
    """
    if not text:
        return text
    char = random.choice(string.ascii_lowercase + string.digits)
    idx = random.randint(0, len(text))
    return text[:idx] + char + text[idx:]


def keyboard_typo(text: str) -> str:
    """
    Заменить случайный символ на соседнюю клавишу.

    Имитирует промах по клавиатуре. Работает и с латиницей,
    и с кириллицей.
    Пример: 'iphone' → 'iohone' (p → o)

    Args:
        text: Исходная строка.

    Returns:
        Строка с заменённым символом.
    """
    if not text:
        return text

    # Объединяем обе раскладки
    all_neighbors = {**_KEYBOARD_NEIGHBORS, **_KEYBOARD_NEIGHBORS_RU}

    candidates = [
        i for i, c in enumerate(text.lower())
        if c in all_neighbors
    ]
    if not candidates:
        return text

    idx = random.choice(candidates)
    neighbors = all_neighbors[text[idx].lower()]
    replacement = random.choice(neighbors)
    return text[:idx] + replacement + text[idx + 1:]


def shuffle_words(text: str) -> str:
    """
    Перемешать порядок слов в строке.

    Имитирует произвольный порядок слов в запросе.
    Пример: 'iphone 13 black' → 'black 13 iphone'

    Args:
        text: Исходная строка.

    Returns:
        Строка с перемешанными словами.
    """
    words = text.split()
    if len(words) < 2:
        return text
    random.shuffle(words)
    return " ".join(words)


def drop_random_word(text: str) -> str:
    """
    Удалить случайное слово из строки.

    Имитирует неполный запрос ('iphone 13 black' → 'iphone 13').
    Не удаляет, если осталось ≤2 слов.

    Args:
        text: Исходная строка.

    Returns:
        Строка без одного случайного слова.
    """
    words = text.split()
    if len(words) <= 2:
        return text
    idx = random.randint(0, len(words) - 1)
    return " ".join(words[:idx] + words[idx + 1:])


def apply_random_noise(text: str, intensity: int = 1) -> str:
    """
    Применить случайную комбинацию зашумлений к тексту.

    Выбирает и применяет N случайных функций зашумления,
    где N = intensity. Чем выше intensity — тем больше
    «повреждений» в тексте.

    Args:
        text: Исходная строка.
        intensity: Количество применяемых зашумлений (1-3).

    Returns:
        Зашумлённая строка.
    """
    noise_functions = [
        swap_adjacent_chars,
        delete_random_char,
        duplicate_random_char,
        insert_random_char,
        keyboard_typo,
    ]

    for _ in range(intensity):
        func = random.choice(noise_functions)
        text = func(text)

    return text
