"""
neuro/model/layers.py
---------------------
Пользовательские слои для Transformer-энкодера.

Содержит:
  - PositionalEncoding: синусоидальное позиционное кодирование,
    добавляет информацию о порядке символов в последовательности.
  - TransformerBlock: один блок энкодера
    (Multi-Head Attention + Feed-Forward Network + Residual + Norm).
"""

import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Синусоидальное позиционное кодирование.

    Добавляет информацию о позиции каждого символа в последовательности.
    Без этого Transformer не различает порядок символов —
    'abc' и 'cba' были бы одинаковыми.

    Позиции кодируются через sin/cos с разными частотами:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """

    def __init__(self, embed_dim: int, max_len: int = 512, **kwargs):
        """
        Args:
            embed_dim: Размерность эмбеддинга.
            max_len: Максимальная длина последовательности.
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.max_len = max_len

        # Вычисляем матрицу позиционного кодирования один раз при создании
        pe = np.zeros((max_len, embed_dim))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pos_encoding = tf.constant(
            pe[np.newaxis, :, :], dtype=tf.float32
        )

    def call(self, x):
        """
        Добавить позиционное кодирование к входному тензору.

        Args:
            x: Тензор shape (batch, seq_len, embed_dim).

        Returns:
            Тензор той же формы с добавленным позиционным кодированием.
        """
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        """Конфигурация для сериализации слоя."""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "max_len": self.max_len,
        })
        return config


class TransformerBlock(tf.keras.layers.Layer):
    """
    Один блок Transformer Encoder.

    Архитектура блока:
        x → MultiHeadAttention → Dropout → Add(residual) → LayerNorm
          → FeedForward(Dense→GELU→Dense) → Dropout → Add(residual) → LayerNorm
          → output

    Residual connections (Add) помогают градиентам проходить через
    глубокую сеть. LayerNorm стабилизирует обучение.
    GELU — активация, которая работает лучше ReLU для Transformer-ов.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            embed_dim: Размерность эмбеддинга.
            num_heads: Количество голов внимания.
            ff_dim: Размер скрытого слоя в FFN.
            dropout: Вероятность dropout.
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        # Multi-Head Self-Attention
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
        )

        # Feed-Forward Network (два Dense слоя с GELU)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="gelu"),
            tf.keras.layers.Dense(embed_dim),
        ])

        # Layer Normalization (стабилизация обучения)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout (регуляризация — случайное «выключение» нейронов)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        """
        Прямой проход через блок.

        Args:
            x: Тензор shape (batch, seq_len, embed_dim).
            training: Флаг режима обучения (для Dropout).

        Returns:
            Тензор той же формы после трансформации.
        """
        # Self-Attention + Residual + Norm
        attn_output = self.attention(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.norm1(x + attn_output)

        # Feed-Forward + Residual + Norm
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.norm2(x + ffn_output)

        return x

    def get_config(self):
        """Конфигурация для сериализации слоя."""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config
