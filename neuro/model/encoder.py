"""
neuro/model/encoder.py
----------------------
Главная модель: ProductEncoder на базе Transformer (PyTorch).

[символы] → Embedding(128) → PositionalEncoding
          → 4 × TransformerBlock(8 heads, ff=512)
          → CLS-pooling → Linear(256) → L2-нормализация
          → вектор 256D
"""

import json
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro.model.layers import PositionalEncoding, TransformerBlock


class ProductEncoder(nn.Module):
    """
    Transformer-энкодер для товарных запросов.

    Принимает последовательность символьных токенов и выдаёт
    нормализованный вектор фиксированной размерности.
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
    ):
        super().__init__()
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

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=max_seq_len)
        self.embed_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.projection = nn.Linear(embed_dim, output_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход: токены → нормализованный вектор.

        Args:
            token_ids: (batch, seq_len) LongTensor.

        Returns:
            (batch, output_dim) — L2-нормализованные векторы.
        """
        x = self.embedding(token_ids)
        x = self.pos_encoding(x)
        x = self.embed_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        # CLS-pooling: вектор первого токена [CLS]
        cls_output = x[:, 0, :]
        output = self.projection(cls_output)
        output = F.normalize(output, p=2, dim=-1)
        return output

    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        tokenizer,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Тексты → numpy массив нормализованных векторов.
        """
        self.eval()
        device = next(self.parameters()).device
        all_vectors = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            token_ids = tokenizer.encode_batch(batch_texts)
            token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
            vectors = self(token_ids).cpu().numpy()
            all_vectors.append(vectors)

        return np.concatenate(all_vectors, axis=0)

    def save_all(self, directory: str) -> None:
        """Сохранить веса + конфиг."""
        os.makedirs(directory, exist_ok=True)
        weights_path = os.path.join(directory, "model.pt")
        torch.save(self.state_dict(), weights_path)

        config_path = os.path.join(directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2)

    @classmethod
    def load_all(cls, directory: str, device: str = "cpu") -> "ProductEncoder":
        """Загрузить модель из директории."""
        config_path = os.path.join(directory, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model = cls(**config)
        weights_path = os.path.join(directory, "model.pt")
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
        model.to(device)
        model.eval()
        return model
