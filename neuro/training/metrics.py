"""
neuro/training/metrics.py
-------------------------
Метрики качества: Recall@K, MRR, Precision@K.

Работают на numpy — не зависят от фреймворка.
"""

import numpy as np


def recall_at_k(
    query_vectors: np.ndarray,
    query_labels: np.ndarray,
    index_vectors: np.ndarray,
    index_labels: np.ndarray,
    k: int = 5,
) -> float:
    similarities = np.dot(query_vectors, index_vectors.T)
    hits = 0
    for i in range(len(query_vectors)):
        top_k_indices = np.argsort(similarities[i])[::-1][:k]
        if query_labels[i] in index_labels[top_k_indices]:
            hits += 1
    return hits / len(query_vectors)


def mean_reciprocal_rank(
    query_vectors: np.ndarray,
    query_labels: np.ndarray,
    index_vectors: np.ndarray,
    index_labels: np.ndarray,
) -> float:
    similarities = np.dot(query_vectors, index_vectors.T)
    rr = []
    for i in range(len(query_vectors)):
        sorted_labels = index_labels[np.argsort(similarities[i])[::-1]]
        positions = np.where(sorted_labels == query_labels[i])[0]
        rr.append(1.0 / (positions[0] + 1) if len(positions) > 0 else 0.0)
    return float(np.mean(rr))


def precision_at_k(
    query_vectors: np.ndarray,
    query_labels: np.ndarray,
    index_vectors: np.ndarray,
    index_labels: np.ndarray,
    k: int = 5,
) -> float:
    similarities = np.dot(query_vectors, index_vectors.T)
    precisions = []
    for i in range(len(query_vectors)):
        top_k_indices = np.argsort(similarities[i])[::-1][:k]
        correct = np.sum(index_labels[top_k_indices] == query_labels[i])
        precisions.append(correct / k)
    return float(np.mean(precisions))
