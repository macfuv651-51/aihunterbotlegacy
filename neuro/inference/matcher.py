"""
neuro/inference/matcher.py
--------------------------
Высокоуровневый API для матчинга товаров.

Объединяет модель (ProductEncoder) и индекс (FAISS) в единый интерфейс.
Один вызов match(query) → список товаров с оценками уверенности.

Это основной класс для интеграции в бота:
    matcher = ProductMatcher(weights_dir, index_dir)
    results = matcher.match("айфон 13 128 черный")
    if matcher.is_confident(results) == "auto":
        send_price(results[0].product["price"])

Уровни уверенности:
    score ≥ 0.92  →  "auto"    (автоматическая отправка)
    score ≥ 0.75  →  "review"  (на проверку админу)
    score < 0.75  →  "reject"  (товар не найден)
"""

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from neuro.inference.index import FAISSIndex
from neuro.model.encoder import ProductEncoder
from neuro.tokenizer.char_tokenizer import CharTokenizer


@dataclass
class MatchResult:
    """
    Результат матчинга одного товара.

    Attributes:
        product: Словарь с данными товара (name, price, category и т.д.).
        score: Оценка уверенности (-1.0 … 1.0).
               Чем ближе к 1.0 — тем увереннее совпадение.
    """

    product: Dict
    score: float


class ProductMatcher:
    """
    Главный класс для поиска товаров по текстовому запросу.

    Загружает обученную модель и FAISS индекс при инициализации.
    Далее каждый вызов match() занимает < 5ms.
    """

    def __init__(
        self,
        weights_dir: str,
        index_dir: str,
        dimension: int = 256,
    ):
        """
        Загрузить модель и индекс из указанных директорий.

        Args:
            weights_dir: Директория с весами модели и токенайзером.
            index_dir: Директория с FAISS индексом.
            dimension: Размерность вектора (должна совпадать с моделью).
        """
        # Загружаем токенайзер
        tokenizer_path = os.path.join(weights_dir, "tokenizer.json")
        self.tokenizer = CharTokenizer.load(tokenizer_path)

        # Загружаем обученную модель
        self.encoder = ProductEncoder.load_all(weights_dir)

        # Загружаем FAISS индекс
        self.index = FAISSIndex.load(index_dir, dimension=dimension)

    def match(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[MatchResult]:
        """
        Найти товары, наиболее похожие на текстовый запрос.

        Args:
            query: Текст запроса (например, 'айфон 13 128 черный').
            top_k: Количество результатов.

        Returns:
            Список MatchResult, отсортированный по убыванию score.
        """
        # Текст → токены → вектор
        token_ids = self.tokenizer.encode(query)
        token_ids = np.expand_dims(token_ids, axis=0)
        query_vector = self.encoder(
            token_ids, training=False
        ).numpy()[0]

        # Вектор → поиск ближайших в FAISS
        results = self.index.search(query_vector, top_k=top_k)

        return [
            MatchResult(product=product, score=score)
            for product, score in results
        ]

    def match_batch(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> List[List[MatchResult]]:
        """
        Батчевый матчинг нескольких запросов за один вызов.

        Args:
            queries: Список текстовых запросов.
            top_k: Количество результатов на запрос.

        Returns:
            Список списков MatchResult.
        """
        token_ids = self.tokenizer.encode_batch(queries)
        vectors = self.encoder(
            token_ids, training=False
        ).numpy()

        all_results = self.index.search_batch(vectors, top_k=top_k)

        return [
            [MatchResult(product=p, score=s) for p, s in results]
            for results in all_results
        ]

    def is_confident(
        self,
        results: List[MatchResult],
        auto_threshold: float = 0.92,
        review_threshold: float = 0.75,
    ) -> str:
        """
        Определить уровень уверенности в лучшем результате.

        Args:
            results: Результаты match().
            auto_threshold: Порог для автоматической отправки.
            review_threshold: Порог для отправки на review.

        Returns:
            'auto'   — автоматическая отправка (score ≥ auto_threshold).
            'review' — нужно подтверждение админа.
            'reject' — товар не найден (score < review_threshold).
        """
        if not results:
            return "reject"

        top_score = results[0].score

        if top_score >= auto_threshold:
            return "auto"
        elif top_score >= review_threshold:
            return "review"
        else:
            return "reject"

    def rebuild_index(
        self,
        products: List[Dict],
        index_dir: str,
    ) -> None:
        """
        Перестроить FAISS индекс из нового списка товаров.

        Вызывается после загрузки нового прайс-листа.
        Все товары кодируются моделью и сохраняются в индекс.
        Занимает ~5 секунд на 1000 товаров.

        Args:
            products: Список словарей из products.json.
            index_dir: Директория для сохранения нового индекса.
        """
        names = [p.get("name", "").lower() for p in products]

        # Кодируем все имена товаров в векторы
        vectors = self.encoder.encode_texts(names, self.tokenizer)
        vectors = vectors.astype(np.float32)

        # Строим новый индекс
        self.index.build(vectors, products)
        self.index.save(index_dir)
