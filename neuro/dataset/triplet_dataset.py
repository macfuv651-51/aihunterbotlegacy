"""
neuro/dataset/triplet_dataset.py
---------------------------------
PyTorch Dataset/DataLoader для обучения на Triplet Loss.
"""

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TripletDataset(Dataset):
    """
    PyTorch Dataset для троек (anchor, positive, negative).

    Все тексты предварительно токенизируются и хранятся как numpy-массивы.
    """

    def __init__(self, anchor_ids, positive_ids, negative_ids):
        self.anchor_ids = anchor_ids
        self.positive_ids = positive_ids
        self.negative_ids = negative_ids

    def __len__(self):
        return len(self.anchor_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.anchor_ids[idx], dtype=torch.long),
            torch.tensor(self.positive_ids[idx], dtype=torch.long),
            torch.tensor(self.negative_ids[idx], dtype=torch.long),
        )


def create_triplet_dataloader(
    triplets: List[Tuple[str, str, str]],
    tokenizer,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Создать PyTorch DataLoader из списка троек.
    """
    anchors = [t[0] for t in triplets]
    positives = [t[1] for t in triplets]
    negatives = [t[2] for t in triplets]

    anchor_ids = tokenizer.encode_batch(anchors)
    positive_ids = tokenizer.encode_batch(positives)
    negative_ids = tokenizer.encode_batch(negatives)

    dataset = TripletDataset(anchor_ids, positive_ids, negative_ids)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )


def mine_hard_negatives(
    product_variants: Dict[int, List[str]],
    encoder,
    tokenizer,
    negatives_per_anchor: int = 5,
) -> List[Tuple[str, str, str]]:
    """
    Найти «трудные» негативные примеры с помощью модели.
    """
    all_texts = []
    all_labels = []
    for label, variants in product_variants.items():
        for v in variants:
            all_texts.append(v)
            all_labels.append(label)

    all_vectors = encoder.encode_texts(all_texts, tokenizer)
    all_labels = np.array(all_labels)

    triplets = []
    for i in range(len(all_texts)):
        anchor_label = all_labels[i]
        similarities = np.dot(all_vectors, all_vectors[i])

        different_mask = all_labels != anchor_label
        same_mask = (all_labels == anchor_label) & (np.arange(len(all_texts)) != i)

        if not np.any(same_mask) or not np.any(different_mask):
            continue

        pos_idx = np.random.choice(np.where(same_mask)[0])

        neg_similarities = similarities.copy()
        neg_similarities[~different_mask] = -2.0
        hard_neg_indices = np.argsort(neg_similarities)[::-1][:negatives_per_anchor]

        for neg_idx in hard_neg_indices:
            triplets.append((all_texts[i], all_texts[pos_idx], all_texts[neg_idx]))

    random.shuffle(triplets)
    return triplets
