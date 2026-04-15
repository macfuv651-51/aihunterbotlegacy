"""
neuro/dataset/triplet_dataset.py
---------------------------------
Генератор батчей для обучения на Triplet Loss.

Создаёт tf.data.Dataset из списка троек (anchor, positive, negative),
кодирует тексты через токенайзер и формирует батчи оптимального
размера для обучения.

Также содержит функцию hard negative mining — поиск «трудных»
негативных примеров, которые ускоряют обучение.
"""

import random
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


def create_triplet_dataset(
    triplets: List[Tuple[str, str, str]],
    tokenizer,
    batch_size: int = 256,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Создать tf.data.Dataset из списка троек.

    Каждая тройка (anchor, positive, negative) кодируется
    через токенайзер в числовые последовательности и упаковывается
    в батчи для обучения.

    Args:
        triplets: Список (anchor_text, positive_text, negative_text).
        tokenizer: Экземпляр CharTokenizer.
        batch_size: Размер батча.
        shuffle: Перемешивать ли данные перед каждой эпохой.

    Returns:
        tf.data.Dataset, выдающий (anchor_ids, positive_ids, negative_ids).
    """
    anchors = [t[0] for t in triplets]
    positives = [t[1] for t in triplets]
    negatives = [t[2] for t in triplets]

    # Кодируем все тексты в числовые последовательности
    anchor_ids = tokenizer.encode_batch(anchors)
    positive_ids = tokenizer.encode_batch(positives)
    negative_ids = tokenizer.encode_batch(negatives)

    dataset = tf.data.Dataset.from_tensor_slices((
        anchor_ids,
        positive_ids,
        negative_ids,
    ))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(triplets), 100_000)
        )

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def mine_hard_negatives(
    product_variants: Dict[int, List[str]],
    encoder,
    tokenizer,
    negatives_per_anchor: int = 5,
) -> List[Tuple[str, str, str]]:
    """
    Найти «трудные» негативные примеры с помощью модели.

    Hard negatives — это товары, которые модель ОШИБОЧНО считает
    похожими на anchor. Обучение на таких примерах заставляет
    модель лучше различать похожие, но разные товары
    (например, iPad Air 11 vs iPad 11 A16).

    Алгоритм:
    1. Кодируем все варианты всех товаров в векторы.
    2. Для каждого anchor ищем ближайшие векторы ДРУГИХ товаров.
    3. Формируем тройки (anchor, positive, hard_negative).

    Рекомендуется запускать после нескольких эпох обучения,
    когда модель уже начала что-то понимать.

    Args:
        product_variants: Словарь {product_idx: [variant1, variant2, ...]}.
        encoder: Обученная (или частично обученная) модель ProductEncoder.
        tokenizer: CharTokenizer.
        negatives_per_anchor: Количество hard negatives на один anchor.

    Returns:
        Список троек (anchor, positive, hard_negative).
    """
    # Собираем все тексты с метками принадлежности к товару
    all_texts = []
    all_labels = []
    for label, variants in product_variants.items():
        for v in variants:
            all_texts.append(v)
            all_labels.append(label)

    # Кодируем всё в векторы через модель
    all_vectors = encoder.encode_texts(all_texts, tokenizer)
    all_labels = np.array(all_labels)

    triplets = []

    for i in range(len(all_texts)):
        anchor_vec = all_vectors[i]
        anchor_label = all_labels[i]

        # Косинусное сходство со всеми остальными
        similarities = np.dot(all_vectors, anchor_vec)

        # Маска: другие товары (для negative)
        different_mask = all_labels != anchor_label

        # Маска: тот же товар, но другой вариант (для positive)
        same_mask = (
            (all_labels == anchor_label)
            & (np.arange(len(all_texts)) != i)
        )

        if not np.any(same_mask) or not np.any(different_mask):
            continue

        # Positive: случайный вариант того же товара
        same_indices = np.where(same_mask)[0]
        pos_idx = np.random.choice(same_indices)

        # Hard negative: самые похожие вектора ДРУГОГО товара
        neg_similarities = similarities.copy()
        neg_similarities[~different_mask] = -2.0
        hard_neg_indices = np.argsort(neg_similarities)[::-1][
            :negatives_per_anchor
        ]

        for neg_idx in hard_neg_indices:
            triplets.append((
                all_texts[i],
                all_texts[pos_idx],
                all_texts[neg_idx],
            ))

    random.shuffle(triplets)
    return triplets
