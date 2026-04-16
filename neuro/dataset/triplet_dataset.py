"""
neuro/dataset/triplet_dataset.py
---------------------------------
PyTorch Dataset/DataLoader for Triplet Loss training.

Two modes:
  1. TripletDataset — pre-tokenised triplets (small catalogs, ≤100K triplets).
  2. OnlineTripletDataset — samples triplets on-the-fly from
     pre-tokenized variants per product  (large catalogs, millions of triplets,
     fits in ~300 MB RAM instead of 4+ GB).
"""

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ═══════════════════════════════════════════════════════════════════════════════
#  OFFLINE — pre-tokenised triplets (legacy, small datasets)
# ═══════════════════════════════════════════════════════════════════════════════

class TripletDataset(Dataset):
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
    anchors   = [t[0] for t in triplets]
    positives = [t[1] for t in triplets]
    negatives = [t[2] for t in triplets]

    anchor_ids   = tokenizer.encode_batch(anchors)
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


# ═══════════════════════════════════════════════════════════════════════════════
#  ONLINE — sample triplets on-the-fly  (large catalogs, 2-3 M+ per epoch)
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineTripletDataset(Dataset):
    """
    Memory-efficient triplet sampler.

    Stores pre-tokenised *variants* per product (~300 MB for 3 M variants)
    instead of all triplets (~4+ GB).  Each __getitem__ call samples a
    fresh random (anchor, positive, negative) triple, so every epoch
    sees completely new combinations.
    """

    def __init__(
        self,
        tokenized_variants: Dict[int, np.ndarray],
        dataset_size: int,
    ):
        self.tokenized_variants = tokenized_variants
        self.product_indices = list(tokenized_variants.keys())
        self.dataset_size = dataset_size
        self._n = len(self.product_indices)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # deterministic product for this idx, random pair within it
        prod_pos = idx % self._n
        prod_idx = self.product_indices[prod_pos]
        variants = self.tokenized_variants[prod_idx]
        n_var = len(variants)

        # anchor & positive — two different variants of the SAME product
        i = random.randint(0, n_var - 1)
        j = random.randint(0, n_var - 2)
        if j >= i:
            j += 1

        anchor   = torch.tensor(variants[i], dtype=torch.long)
        positive = torch.tensor(variants[j], dtype=torch.long)

        # negative — a variant of a DIFFERENT product
        neg_pos = random.randint(0, self._n - 2)
        if neg_pos >= prod_pos:
            neg_pos += 1
        neg_variants = self.tokenized_variants[self.product_indices[neg_pos]]
        k = random.randint(0, len(neg_variants) - 1)
        negative = torch.tensor(neg_variants[k], dtype=torch.long)

        return anchor, positive, negative


def _worker_init_fn(worker_id):
    """Ensure each DataLoader worker gets a unique random seed."""
    seed = torch.initial_seed() % (2**32) + worker_id
    random.seed(seed)
    np.random.seed(seed % (2**32))
