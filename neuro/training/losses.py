"""
neuro/training/losses.py
------------------------
Функции потерь для обучения Siamese-энкодера.

Triplet Loss: заставляет модель «притягивать» anchor к positive
и «отталкивать» от negative в векторном пространстве.

Формула:
    Loss = max(0, dist(anchor, positive) - dist(anchor, negative) + margin)

Если positive ближе к anchor, чем negative (с запасом margin), loss = 0.
Иначе модель штрафуется пропорционально разнице расстояний.
"""

import tensorflow as tf


def triplet_loss(
    anchor: tf.Tensor,
    positive: tf.Tensor,
    negative: tf.Tensor,
    margin: float = 0.3,
) -> tf.Tensor:
    """
    Вычислить Triplet Margin Loss.

    Для L2-нормализованных векторов:
        dist²(a, b) = 2 - 2·dot(a, b) = sum((a - b)²)

    Args:
        anchor: Вектор якоря, shape (batch, dim).
        positive: Вектор позитива (тот же товар), shape (batch, dim).
        negative: Вектор негатива (другой товар), shape (batch, dim).
        margin: Минимальный запас расстояния между pos и neg.

    Returns:
        Скалярный loss (среднее по батчу).
    """
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(loss)


def online_triplet_loss(
    embeddings: tf.Tensor,
    labels: tf.Tensor,
    margin: float = 0.3,
) -> tf.Tensor:
    """
    Online Triplet Loss с автоматическим подбором троек внутри батча.

    Вместо заранее подготовленных троек — берёт батч эмбеддингов
    с метками классов и находит все валидные тройки на лету.

    Использует semi-hard стратегию: негатив дальше позитива,
    но ближе чем позитив + margin. Это оптимальный режим обучения:
    не слишком лёгкий (модель уже справляется), но и не слишком
    сложный (модель ещё не может различить).

    Args:
        embeddings: Тензор shape (batch, dim).
        labels: Тензор shape (batch,) с метками классов (product_id).
        margin: Margin для triplet loss.

    Returns:
        Скалярный loss.
    """
    # Матрица всех попарных расстояний
    pairwise_dist = _pairwise_distances(embeddings)

    # Маски: одинаковые метки (один товар) / разные метки
    labels_equal = tf.equal(
        tf.expand_dims(labels, 0),
        tf.expand_dims(labels, 1),
    )
    labels_not_equal = tf.logical_not(labels_equal)

    # Расстояния anchor-positive и anchor-negative для всех троек
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

    # Loss для всех возможных троек
    triplet_loss_val = anchor_positive_dist - anchor_negative_dist + margin

    # Маска валидных троек: (a, p) — одного класса, (a, n) — разных
    mask_anchor_positive = tf.expand_dims(labels_equal, 2)
    mask_anchor_negative = tf.expand_dims(labels_not_equal, 1)

    # Исключаем тривиальные тройки где a == p или a == n
    indices_not_equal = tf.logical_not(
        tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    )
    mask_not_self_a_p = tf.expand_dims(indices_not_equal, 2)
    mask_not_self_a_n = tf.expand_dims(indices_not_equal, 1)

    mask = tf.cast(
        mask_anchor_positive
        & mask_anchor_negative
        & mask_not_self_a_p
        & mask_not_self_a_n,
        tf.float32,
    )

    # Оставляем только тройки с положительным loss (semi-hard)
    triplet_loss_val = tf.maximum(triplet_loss_val * mask, 0.0)

    # Усредняем по ненулевым тройкам
    num_positive_triplets = tf.reduce_sum(
        tf.cast(triplet_loss_val > 1e-16, tf.float32)
    )
    loss = tf.reduce_sum(triplet_loss_val) / (num_positive_triplets + 1e-16)

    return loss


def _pairwise_distances(embeddings: tf.Tensor) -> tf.Tensor:
    """
    Вычислить матрицу попарных евклидовых расстояний.

    Формула: ||a - b||² = ||a||² + ||b||² - 2·a·b

    Args:
        embeddings: Тензор shape (batch, dim).

    Returns:
        Тензор shape (batch, batch) с расстояниями.
    """
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.linalg.diag_part(dot_product)

    distances = (
        tf.expand_dims(square_norm, 0)
        - 2.0 * dot_product
        + tf.expand_dims(square_norm, 1)
    )

    # Обнуляем отрицательные значения (артефакты float-арифметики)
    distances = tf.maximum(distances, 0.0)
    return distances
