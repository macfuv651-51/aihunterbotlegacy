"""
neuro/training/trainer.py
-------------------------
Цикл обучения нейросети с Triplet Loss (PyTorch).

Управляет:
- Обучением модели (train loop)
- Валидацией (Recall@K, MRR на каждой эпохе)
- Learning rate scheduling (cosine annealing)
- Сохранением лучших весов (checkpointing по Recall@1)
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuro.training.losses import triplet_loss
from neuro.training.metrics import mean_reciprocal_rank, recall_at_k


class Trainer:
    """
    Полный цикл обучения ProductEncoder на PyTorch.
    """

    def __init__(
        self,
        model,
        learning_rate: float = 3e-4,
        min_lr: float = 1e-5,
        margin: float = 0.3,
        checkpoint_dir: str = "weights",
        device: str = "auto",
    ):
        self.model = model
        self.margin = margin
        self.checkpoint_dir = checkpoint_dir
        self.best_recall = 0.0
        self.history: List[Dict] = []

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.learning_rate = learning_rate
        self.min_lr = min_lr

    def train(
        self,
        train_loader: DataLoader,
        val_data: Optional[Tuple] = None,
        epochs: int = 100,
        tokenizer=None,
    ) -> List[Dict]:
        """
        Полный цикл обучения.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=self.min_lr
        )

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            self.model.train()
            epoch_losses = []

            for anchor_ids, positive_ids, negative_ids in train_loader:
                anchor_ids = anchor_ids.to(self.device)
                positive_ids = positive_ids.to(self.device)
                negative_ids = negative_ids.to(self.device)

                optimizer.zero_grad()

                anchor_emb = self.model(anchor_ids)
                positive_emb = self.model(positive_ids)
                negative_emb = self.model(negative_ids)

                loss = triplet_loss(
                    anchor_emb, positive_emb, negative_emb, self.margin
                )

                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()

            avg_loss = float(np.mean(epoch_losses))
            elapsed = time.time() - epoch_start
            lr = optimizer.param_groups[0]["lr"]

            metrics = {
                "epoch": epoch,
                "loss": avg_loss,
                "time": elapsed,
                "lr": lr,
            }

            if val_data is not None and tokenizer is not None:
                val_metrics = self._evaluate(val_data, tokenizer)
                metrics.update(val_metrics)

                if val_metrics.get("recall@1", 0) > self.best_recall:
                    self.best_recall = val_metrics["recall@1"]
                    self.model.save_all(self.checkpoint_dir)
                    metrics["saved"] = True

            self.history.append(metrics)
            self._log_epoch(metrics)

        return self.history

    def _evaluate(self, val_data: Tuple, tokenizer) -> Dict[str, float]:
        val_texts, val_labels, index_texts, index_labels = val_data

        val_vectors = self.model.encode_texts(val_texts, tokenizer)
        index_vectors = self.model.encode_texts(index_texts, tokenizer)

        val_labels_arr = np.array(val_labels)
        index_labels_arr = np.array(index_labels)

        r1 = recall_at_k(val_vectors, val_labels_arr, index_vectors, index_labels_arr, k=1)
        r5 = recall_at_k(val_vectors, val_labels_arr, index_vectors, index_labels_arr, k=5)
        mrr = mean_reciprocal_rank(val_vectors, val_labels_arr, index_vectors, index_labels_arr)

        return {"recall@1": r1, "recall@5": r5, "mrr": mrr}

    def _log_epoch(self, metrics: Dict) -> None:
        parts = [
            f"Epoch {metrics['epoch']:3d}",
            f"loss={metrics['loss']:.4f}",
            f"lr={metrics['lr']:.2e}",
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

        print(" | ".join(parts), flush=True)
