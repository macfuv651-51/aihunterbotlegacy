"""
neuro/training/trainer.py
-------------------------
Цикл обучения нейросети с Triplet Loss.

Управляет:
- Обучением модели (train loop с @tf.function для скорости)
- Валидацией (Recall@K, MRR на каждой эпохе)
- Learning rate scheduling (cosine decay: 3e-4 → 1e-5)
- Сохранением лучших весов (checkpointing по Recall@1)
- Логированием метрик в консоль

Типичный запуск:
    trainer = Trainer(model, learning_rate=3e-4)
    history = trainer.train(dataset, val_data, epochs=100, tokenizer=tok)
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from neuro.training.losses import triplet_loss
from neuro.training.metrics import mean_reciprocal_rank, recall_at_k


class Trainer:
    """
    Класс-обёртка для полного цикла обучения ProductEncoder.

    Автоматически:
    - Настраивает cosine decay для learning rate
    - Считает Recall@1, Recall@5, MRR на валидации
    - Сохраняет лучшую модель (по Recall@1)
    - Логирует прогресс в консоль
    """

    def __init__(
        self,
        model,
        learning_rate: float = 3e-4,
        min_lr: float = 1e-5,
        margin: float = 0.3,
        checkpoint_dir: str = "weights",
    ):
        """
        Args:
            model: Экземпляр ProductEncoder.
            learning_rate: Начальный learning rate.
            min_lr: Минимальный learning rate (cosine decay).
            margin: Margin для Triplet Loss.
            checkpoint_dir: Директория для сохранения лучших весов.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.margin = margin
        self.checkpoint_dir = checkpoint_dir
        self.best_recall = 0.0
        self.history: List[Dict] = []

    def _build_optimizer(
        self, total_steps: int
    ) -> tf.keras.optimizers.Optimizer:
        """
        Создать Adam оптимизатор с cosine decay расписанием.

        Learning rate плавно снижается от learning_rate до min_lr
        по косинусной кривой. Это стандартная практика для Transformer:
        высокий lr в начале (быстрое обучение), низкий в конце
        (тонкая настройка).

        Args:
            total_steps: Общее количество шагов обучения (steps × epochs).

        Returns:
            Настроенный Adam оптимизатор.
        """
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=total_steps,
            alpha=self.min_lr / self.learning_rate,
        )
        return tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    @tf.function
    def _train_step(
        self,
        anchor_ids: tf.Tensor,
        positive_ids: tf.Tensor,
        negative_ids: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer,
    ) -> tf.Tensor:
        """
        Один шаг обучения: forward → loss → backward → update.

        @tf.function компилирует этот метод в граф TensorFlow
        для максимальной скорости на CPU.

        Args:
            anchor_ids: Токены якорей, shape (batch, seq_len).
            positive_ids: Токены позитивов, shape (batch, seq_len).
            negative_ids: Токены негативов, shape (batch, seq_len).
            optimizer: Оптимизатор с lr schedule.

        Returns:
            Значение loss для этого батча.
        """
        with tf.GradientTape() as tape:
            anchor_emb = self.model(anchor_ids, training=True)
            positive_emb = self.model(positive_ids, training=True)
            negative_emb = self.model(negative_ids, training=True)

            loss = triplet_loss(
                anchor_emb, positive_emb, negative_emb, self.margin
            )

        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        return loss

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_data: Optional[Tuple] = None,
        epochs: int = 100,
        tokenizer=None,
    ) -> List[Dict]:
        """
        Полный цикл обучения.

        На каждой эпохе:
        1. Прогонять все батчи через _train_step
        2. Считать средний loss
        3. (Опционально) валидация — Recall@1, Recall@5, MRR
        4. Сохранить модель если Recall@1 улучшился

        Args:
            train_dataset: tf.data.Dataset с тройками (anchor, pos, neg).
            val_data: (val_texts, val_labels, index_texts, index_labels)
                      для Recall@K валидации. None = без валидации.
            epochs: Количество эпох.
            tokenizer: CharTokenizer (нужен для валидации).

        Returns:
            История обучения: список словарей
            {epoch, loss, recall@1, recall@5, mrr}.
        """
        # Считаем шаги для cosine decay schedule
        steps_per_epoch = sum(1 for _ in train_dataset)
        total_steps = steps_per_epoch * epochs
        optimizer = self._build_optimizer(total_steps)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            epoch_losses = []

            # Тренировка: проход по всем батчам
            for batch in train_dataset:
                anchor_ids, positive_ids, negative_ids = batch
                loss = self._train_step(
                    anchor_ids, positive_ids, negative_ids, optimizer
                )
                epoch_losses.append(float(loss))

            avg_loss = float(np.mean(epoch_losses))
            elapsed = time.time() - epoch_start

            # Валидация (если данные предоставлены)
            metrics = {
                "epoch": epoch,
                "loss": avg_loss,
                "time": elapsed,
            }

            if val_data is not None and tokenizer is not None:
                val_metrics = self._evaluate(val_data, tokenizer)
                metrics.update(val_metrics)

                # Сохраняем лучшую модель по Recall@1
                if val_metrics.get("recall@1", 0) > self.best_recall:
                    self.best_recall = val_metrics["recall@1"]
                    self.model.save_all(self.checkpoint_dir)
                    metrics["saved"] = True

            self.history.append(metrics)
            self._log_epoch(metrics)

        return self.history

    def _evaluate(
        self,
        val_data: Tuple,
        tokenizer,
    ) -> Dict[str, float]:
        """
        Вычислить метрики на валидационной выборке.

        Args:
            val_data: (val_texts, val_labels, index_texts, index_labels).
            tokenizer: CharTokenizer.

        Returns:
            Словарь метрик {recall@1, recall@5, mrr}.
        """
        val_texts, val_labels, index_texts, index_labels = val_data

        # Кодируем валидационные запросы и эталоны
        val_vectors = self.model.encode_texts(val_texts, tokenizer)
        index_vectors = self.model.encode_texts(index_texts, tokenizer)

        val_labels_arr = np.array(val_labels)
        index_labels_arr = np.array(index_labels)

        r1 = recall_at_k(
            val_vectors, val_labels_arr,
            index_vectors, index_labels_arr, k=1,
        )
        r5 = recall_at_k(
            val_vectors, val_labels_arr,
            index_vectors, index_labels_arr, k=5,
        )
        mrr = mean_reciprocal_rank(
            val_vectors, val_labels_arr,
            index_vectors, index_labels_arr,
        )

        return {"recall@1": r1, "recall@5": r5, "mrr": mrr}

    def _log_epoch(self, metrics: Dict) -> None:
        """
        Вывести метрики эпохи в консоль (одна строка).

        Args:
            metrics: Словарь с метриками текущей эпохи.
        """
        parts = [
            f"Epoch {metrics['epoch']:3d}",
            f"loss={metrics['loss']:.4f}",
            f"time={metrics['time']:.1f}s",
        ]

        if "recall@1" in metrics:
            parts.append(f"R@1={metrics['recall@1']:.3f}")
        if "recall@5" in metrics:
            parts.append(f"R@5={metrics['recall@5']:.3f}")
        if "mrr" in metrics:
            parts.append(f"MRR={metrics['mrr']:.3f}")
        if metrics.get("saved"):
            parts.append("★ saved")

        print(" | ".join(parts))
