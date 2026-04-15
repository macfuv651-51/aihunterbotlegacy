"""
neuro/training/metrics.py
-------------------------
Метрики качества для оценки нейросети матчинга.

Recall@K: доля запросов, для которых правильный товар
    оказался среди K ближайших результатов.
    R@1 = 0.95 означает: в 95% случаев правильный товар — на первом месте.

MRR (Mean Reciprocal Rank): среднее от 1/позиция_правильного_ответа.
    MRR = 1.0 означает: правильный товар всегда на первом месте.
    MRR = 0.5 означает: в среднем на втором месте.

Precision@K: доля правильных результатов среди K ближайших.
"""

import numpy as np


def recall_at_k(
    query_vectors: np.ndarray,
    query_labels: np.ndarray,
    index_vectors: np.ndarray,
    index_labels: np.ndarray,
    k: int = 5,
) -> float:
    """
    Вычислить Recall@K.

    Для каждого запроса ищем K ближайших векторов в индексе
    и проверяем, есть ли среди них правильный товар.

    Args:
        query_vectors: Векторы запросов, shape (n_queries, dim).
        query_labels: Метки запросов (product_id), shape (n_queries,).
        index_vectors: Векторы индекса (эталоны), shape (n_index, dim).
        index_labels: Метки индекса, shape (n_index,).
        k: Количество ближайших соседей.

    Returns:
        Recall@K (от 0.0 до 1.0).
    """
    # Косинусное сходство (вектора уже L2-нормализованы)
    similarities = np.dot(query_vectors, index_vectors.T)

    hits = 0
    for i in range(len(query_vectors)):
        top_k_indices = np.argsort(similarities[i])[::-1][:k]
        top_k_labels = index_labels[top_k_indices]
        if query_labels[i] in top_k_labels:
            hits += 1

    return hits / len(query_vectors)


def mean_reciprocal_rank(
    query_vectors: np.ndarray,
    query_labels: np.ndarray,
    index_vectors: np.ndarray,
    index_labels: np.ndarray,
) -> float:
    """
    Вычислить Mean Reciprocal Rank (MRR).

    MRR = среднее от 1/позиция_правильного_ответа.
    Чем ближе к 1.0 — тем лучше.

    Args:
        query_vectors: Векторы запросов, shape (n_queries, dim).
        query_labels: Метки запросов, shape (n_queries,).
        index_vectors: Векторы индекса, shape (n_index, dim).
        index_labels: Метки индекса, shape (n_index,).

    Returns:
        MRR (от 0.0 до 1.0).
    """
    similarities = np.dot(query_vectors, index_vectors.T)

    reciprocal_ranks = []
    for i in range(len(query_vectors)):
        sorted_indices = np.argsort(similarities[i])[::-1]
        sorted_labels = index_labels[sorted_indices]
        positions = np.where(sorted_labels == query_labels[i])[0]
        if len(positions) > 0:
            reciprocal_ranks.append(1.0 / (positions[0] + 1))
        else:
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks))


def precision_at_k(
    query_vectors: np.ndarray,
    query_labels: np.ndarray,
    index_vectors: np.ndarray,
    index_labels: np.ndarray,
    k: int = 5,
) -> float:
    """
    Вычислить Precision@K.

    Доля правильных результатов среди K ближайших.

    Args:
        query_vectors: Векторы запросов, shape (n_queries, dim).
        query_labels: Метки запросов, shape (n_queries,).
        index_vectors: Векторы индекса, shape (n_index, dim).
        index_labels: Метки индекса, shape (n_index,).
        k: Количество ближайших соседей.

    Returns:
        Precision@K (от 0.0 до 1.0).
    """
    similarities = np.dot(query_vectors, index_vectors.T)

    precisions = []
    for i in range(len(query_vectors)):
        top_k_indices = np.argsort(similarities[i])[::-1][:k]
        top_k_labels = index_labels[top_k_indices]
        correct = np.sum(top_k_labels == query_labels[i])
        precisions.append(correct / k)

    return float(np.mean(precisions))
