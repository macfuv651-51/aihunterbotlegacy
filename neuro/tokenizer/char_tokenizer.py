"""
neuro/tokenizer/char_tokenizer.py
----------------------------------
Символьный токенайзер для нейросетевого матчинга товаров.

Работает на уровне отдельных символов (букв, цифр, пробелов).
Это даёт устойчивость к опечаткам: 'iphon' и 'iphone' отличаются
всего одним символом, а не целым словом.

Поддерживает латиницу, кириллицу и цифры.

Специальные токены:
    [PAD] = 0  — заполнитель (дополнение до max_len)
    [CLS] = 1  — маркер начала последовательности (его вектор = эмбеддинг текста)
    [UNK] = 2  — неизвестный символ (не встречался при fit)
"""

import json
import os
from typing import Dict, List

import numpy as np


# ─── Специальные токены ───────────────────────────────────────────────────────

PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
UNK_TOKEN = "[UNK]"

PAD_ID = 0
CLS_ID = 1
UNK_ID = 2


class CharTokenizer:
    """
    Символьный токенайзер.

    Превращает текст в последовательность числовых индексов,
    где каждый символ = один токен. Добавляет [CLS] в начало,
    дополняет [PAD] до max_len.
    """

    def __init__(self, max_len: int = 128):
        """
        Инициализация токенайзера.

        Args:
            max_len: Максимальная длина последовательности (символов).
                     Включает [CLS] токен.
        """
        self.max_len = max_len
        self.char_to_id: Dict[str, int] = {
            PAD_TOKEN: PAD_ID,
            CLS_TOKEN: CLS_ID,
            UNK_TOKEN: UNK_ID,
        }
        self.id_to_char: Dict[int, str] = {
            v: k for k, v in self.char_to_id.items()
        }
        self._fitted = False

    @property
    def vocab_size(self) -> int:
        """Размер словаря (количество уникальных символов + спецтокены)."""
        return len(self.char_to_id)

    def fit(self, texts: List[str]) -> "CharTokenizer":
        """
        Построить словарь из набора текстов.

        Проходит по всем символам во всех текстах и присваивает
        каждому уникальному символу числовой индекс.

        Args:
            texts: Список строк для построения словаря.

        Returns:
            self (для цепочки вызовов).
        """
        chars = set()
        for text in texts:
            chars.update(text.lower())

        for char in sorted(chars):
            if char not in self.char_to_id:
                idx = len(self.char_to_id)
                self.char_to_id[char] = idx
                self.id_to_char[idx] = char

        self._fitted = True
        return self

    def encode(self, text: str) -> np.ndarray:
        """
        Закодировать текст в массив числовых индексов.

        Добавляет [CLS] токен в начало. Обрезает или дополняет
        до max_len. Неизвестные символы заменяются на [UNK].

        Args:
            text: Входная строка.

        Returns:
            numpy массив shape (max_len,) с индексами символов.
        """
        text = text.lower()
        tokens = [CLS_ID]

        for char in text[: self.max_len - 1]:
            token_id = self.char_to_id.get(char, UNK_ID)
            tokens.append(token_id)

        # Дополняем до max_len нулями (PAD)
        while len(tokens) < self.max_len:
            tokens.append(PAD_ID)

        return np.array(tokens[: self.max_len], dtype=np.int32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Закодировать пакет текстов.

        Args:
            texts: Список строк.

        Returns:
            numpy массив shape (len(texts), max_len).
        """
        return np.array([self.encode(t) for t in texts], dtype=np.int32)

    def decode(self, token_ids: np.ndarray) -> str:
        """
        Декодировать массив индексов обратно в текст.

        Args:
            token_ids: Массив числовых индексов.

        Returns:
            Декодированная строка (без [PAD] и [CLS]).
        """
        chars = []
        for tid in token_ids:
            tid = int(tid)
            if tid in (PAD_ID, CLS_ID):
                continue
            char = self.id_to_char.get(tid, "?")
            if char == UNK_TOKEN:
                chars.append("?")
            else:
                chars.append(char)
        return "".join(chars)

    def save(self, path: str) -> None:
        """
        Сохранить токенайзер на диск в формате JSON.

        Args:
            path: Путь к файлу для сохранения.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "max_len": self.max_len,
            "char_to_id": self.char_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """
        Загрузить токенайзер из JSON файла.

        Args:
            path: Путь к файлу токенайзера.

        Returns:
            Загруженный экземпляр CharTokenizer.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(max_len=data["max_len"])
        tokenizer.char_to_id = data["char_to_id"]
        tokenizer.id_to_char = {
            int(v): k for k, v in data["char_to_id"].items()
        }
        tokenizer._fitted = True
        return tokenizer
