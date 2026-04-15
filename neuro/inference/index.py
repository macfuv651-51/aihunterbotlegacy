"""
neuro/inference/index.py
------------------------
FAISS индекс для быстрого поиска ближайших товаров.

Хранит векторные представления всех товаров из каталога.
Поиск по миллиону записей — менее 5ms на CPU.

FAISS (Facebook AI Similarity Search) — библиотека Meta
для эффективного поиска ближайших соседей в пространстве
высокой размерности.

Мы используем IndexFlatIP (Inner Product), который для
L2-нормализованных векторов эквивалентен cosine similarity.

Файлы на диске:
  - faiss.index      — бинарный индекс с векторами
  - product_map.json — привязка номера вектора к товару
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class FAISSIndex:
    """
    Обёртка над FAISS для поиска товаров по эмбеддингам.

    Принимает массив нормализованных векторов (от ProductEncoder)
    и обеспечивает мгновенный поиск ближайших соседей.
    """

    def __init__(self, dimension: int = 256):
        """
        Args:
            dimension: Размерность вектора эмбеддинга
                       (должна совпадать с OUTPUT_DIM модели).
        """
        if faiss is None:
            raise ImportError(
                "faiss-cpu не установлен. "
                "Выполните: pip install faiss-cpu"
            )

        self.dimension = dimension
        # IndexFlatIP = Inner Product (cosine для L2-нормализованных)
        self.index = faiss.IndexFlatIP(dimension)
        self.product_map: List[Dict] = []

    @property
    def size(self) -> int:
        """Количество векторов в индексе."""
        return self.index.ntotal

    def build(
        self,
        vectors: np.ndarray,
        products: List[Dict],
    ) -> None:
        """
        Построить индекс из массива векторов.

        Каждый вектор соответствует одному товару из products.
        Порядок векторов = порядку товаров в списке.

        Args:
            vectors: Массив shape (n_products, dimension), dtype float32.
                     Вектора должны быть L2-нормализованы.
            products: Список словарей с данными о товарах
                      (name, price, category и т.д.).

        Raises:
            ValueError: Если размерность не совпадает.
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Размерность векторов ({vectors.shape[1]}) "
                f"не совпадает с индексом ({self.dimension})."
            )

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.reset()
        self.index.add(vectors)
        self.product_map = products

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[Dict, float]]:
        """
        Найти top_k ближайших товаров к вектору запроса.

        Args:
            query_vector: Вектор запроса, shape (dim,) или (1, dim).
            top_k: Количество результатов.

        Returns:
            Список кортежей (product_dict, score),
            отсортированный по убыванию score.
            Score ∈ [-1, 1] — cosine similarity.
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = np.ascontiguousarray(
            query_vector, dtype=np.float32
        )
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.product_map):
                continue
            results.append((self.product_map[idx], float(score)))

        return results

    def search_batch(
        self,
        query_vectors: np.ndarray,
        top_k: int = 5,
    ) -> List[List[Tuple[Dict, float]]]:
        """
        Батчевый поиск для нескольких запросов одновременно.

        Args:
            query_vectors: Массив shape (n_queries, dimension).
            top_k: Количество результатов на запрос.

        Returns:
            Список списков кортежей (product_dict, score).
        """
        query_vectors = np.ascontiguousarray(
            query_vectors, dtype=np.float32
        )
        scores, indices = self.index.search(query_vectors, top_k)

        all_results = []
        for batch_scores, batch_indices in zip(scores, indices):
            results = []
            for score, idx in zip(batch_scores, batch_indices):
                if idx < 0 or idx >= len(self.product_map):
                    continue
                results.append((self.product_map[idx], float(score)))
            all_results.append(results)

        return all_results

    def save(self, directory: str) -> None:
        """
        Сохранить индекс и карту товаров на диск.

        Создаёт:
          - faiss.index      — бинарный FAISS индекс
          - product_map.json — JSON с данными товаров

        Args:
            directory: Директория для сохранения.
        """
        os.makedirs(directory, exist_ok=True)

        index_path = os.path.join(directory, "faiss.index")
        faiss.write_index(self.index, index_path)

        map_path = os.path.join(directory, "product_map.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(
                self.product_map, f, ensure_ascii=False, indent=2
            )

    @classmethod
    def load(cls, directory: str, dimension: int = 256) -> "FAISSIndex":
        """
        Загрузить индекс и карту товаров с диска.

        Args:
            directory: Директория с сохранённым индексом.
            dimension: Размерность вектора.

        Returns:
            Загруженный экземпляр FAISSIndex, готовый к поиску.
        """
        instance = cls(dimension=dimension)

        index_path = os.path.join(directory, "faiss.index")
        instance.index = faiss.read_index(index_path)

        map_path = os.path.join(directory, "product_map.json")
        with open(map_path, "r", encoding="utf-8") as f:
            instance.product_map = json.load(f)

        return instance
