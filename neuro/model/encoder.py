"""
neuro/model/encoder.py
----------------------
Главная модель: ProductEncoder на базе Transformer.

Принимает текст → превращает в вектор фиксированной длины (256-dim).
Два одинаковых товара → близкие векторы.
Два разных товара → далёкие векторы.

Архитектура:
  [символы] → Embedding(128) → PositionalEncoding
            → 4 × TransformerBlock(8 heads, ff=512)
            → CLS-pooling (берём вектор первого токена)
            → Dense(256) → L2-нормализация
            → вектор 256D

CLS-pooling: токен [CLS] (индекс 1) стоит в начале каждой
последовательности. После прохода через Transformer его вектор
аккумулирует информацию обо всей строке. Мы берём именно его
как «смысловой отпечаток» всего запроса.

L2-нормализация: делает все вектора единичной длины, после чего
cosine similarity = просто dot product (быстрее вычислять).
"""

import json
import os
from typing import List, Optional

import numpy as np
import tensorflow as tf

from neuro.model.layers import PositionalEncoding, TransformerBlock


class ProductEncoder(tf.keras.Model):
    """
    Transformer-энкодер для товарных запросов.

    Принимает последовательность символьных токенов и выдаёт
    нормализованный вектор фиксированной размерности.
    Обучается через Triplet Loss — «притягивает» вектора
    одинаковых товаров и «отталкивает» разных.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        num_layers: int = 4,
        output_dim: int = 256,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            vocab_size: Размер словаря токенайзера (кол-во уникальных символов).
            embed_dim: Размерность символьного эмбеддинга.
            num_heads: Количество голов внимания.
            ff_dim: Размер скрытого слоя в FFN.
            num_layers: Количество слоёв Transformer.
            output_dim: Размерность выходного вектора.
            max_seq_len: Максимальная длина входной последовательности.
            dropout: Вероятность dropout.
        """
        super().__init__(**kwargs)

        self._config = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "num_layers": num_layers,
            "output_dim": output_dim,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
        }

        # Символьный эмбеддинг: индекс символа → вектор
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=False,
        )

        # Позиционное кодирование (порядок символов)
        self.pos_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=max_seq_len,
        )

        # Dropout после эмбеддинга
        self.embed_dropout = tf.keras.layers.Dropout(dropout)

        # Стек Transformer блоков (основа модели)
        self.transformer_blocks = [
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                name=f"transformer_block_{i}",
            )
            for i in range(num_layers)
        ]

        # Проекция [CLS]-вектора в выходное пространство
        self.projection = tf.keras.layers.Dense(
            output_dim, name="projection"
        )

    def call(self, token_ids, training=False):
        """
        Прямой проход: токены → нормализованный вектор.

        Args:
            token_ids: Тензор shape (batch, seq_len) с индексами токенов.
            training: Флаг режима обучения (влияет на Dropout).

        Returns:
            Тензор shape (batch, output_dim) — L2-нормализованные векторы.
        """
        # Эмбеддинг: (batch, seq_len) → (batch, seq_len, embed_dim)
        x = self.embedding(token_ids)

        # Позиционное кодирование
        x = self.pos_encoding(x)
        x = self.embed_dropout(x, training=training)

        # Прогон через каждый Transformer блок
        for block in self.transformer_blocks:
            x = block(x, training=training)

        # CLS-pooling: берём вектор первого токена [CLS]
        cls_output = x[:, 0, :]

        # Проекция в выходное пространство
        output = self.projection(cls_output)

        # L2-нормализация: все вектора становятся единичной длины
        output = tf.nn.l2_normalize(output, axis=-1)

        return output

    def encode_texts(
        self,
        texts: List[str],
        tokenizer,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Вспомогательный метод: тексты → массив нормализованных векторов.

        Удобно для построения FAISS индекса и валидации.
        Разбивает тексты на батчи для экономии памяти.

        Args:
            texts: Список строк для энкодинга.
            tokenizer: Экземпляр CharTokenizer.
            batch_size: Размер батча для инференса.

        Returns:
            numpy массив shape (len(texts), output_dim).
        """
        all_vectors = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            token_ids = tokenizer.encode_batch(batch_texts)
            vectors = self(token_ids, training=False).numpy()
            all_vectors.append(vectors)

        return np.concatenate(all_vectors, axis=0)

    def get_config(self):
        """Конфигурация модели для сериализации."""
        return dict(self._config)

    def save_all(self, directory: str) -> None:
        """
        Сохранить модель целиком: веса + конфигурацию.

        Создаёт в directory:
          - model.weights.h5 — бинарные веса (числа)
          - config.json — архитектура (кол-во слоёв, размерности и т.д.)

        Args:
            directory: Директория для сохранения.
        """
        os.makedirs(directory, exist_ok=True)

        # Сохраняем веса
        weights_path = os.path.join(directory, "model.weights.h5")
        self.save_weights(weights_path)

        # Сохраняем конфиг архитектуры
        config_path = os.path.join(directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2)

    @classmethod
    def load_all(cls, directory: str) -> "ProductEncoder":
        """
        Загрузить модель из директории: конфиг + веса.

        Args:
            directory: Директория с сохранённой моделью.

        Returns:
            Загруженный и готовый к инференсу ProductEncoder.
        """
        # Читаем конфиг
        config_path = os.path.join(directory, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Создаём модель с той же архитектурой
        model = cls(**config)

        # Инициализируем веса одним прямым проходом (TF требует это)
        dummy_input = tf.zeros(
            (1, config["max_seq_len"]), dtype=tf.int32
        )
        model(dummy_input)

        # Загружаем обученные веса
        weights_path = os.path.join(directory, "model.weights.h5")
        model.load_weights(weights_path)

        return model
