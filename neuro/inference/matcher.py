"""
neuro/inference/matcher.py
--------------------------
Высокоуровневый API для матчинга товаров (PyTorch).

    matcher = ProductMatcher(weights_dir, index_dir)
    results = matcher.match("айфон 13 128 черный")
    if matcher.is_confident(results) == "auto":
        send_price(results[0].product["price"])
"""

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from neuro.inference.index import FAISSIndex
from neuro.model.encoder import ProductEncoder
from neuro.tokenizer.char_tokenizer import CharTokenizer


@dataclass
class MatchResult:
    product: Dict
    score: float


class ProductMatcher:
    def __init__(
        self,
        weights_dir: str,
        index_dir: str,
        dimension: int = 256,
        device: str = "cpu",
    ):
        tokenizer_path = os.path.join(weights_dir, "tokenizer.json")
        self.tokenizer = CharTokenizer.load(tokenizer_path)
        self.encoder = ProductEncoder.load_all(weights_dir, device=device)
        self.index = FAISSIndex.load(index_dir, dimension=dimension)
        self.device = torch.device(device)

    def match(self, query: str, top_k: int = 5) -> List[MatchResult]:
        encoded = self.tokenizer.encode(query)
        token_ids = torch.tensor(
            encoded.reshape(1, -1),
            dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            query_vector = self.encoder(token_ids).cpu().numpy()[0]

        results = self.index.search(query_vector, top_k=top_k)
        return [MatchResult(product=p, score=s) for p, s in results]

    def match_batch(self, queries: List[str], top_k: int = 5) -> List[List[MatchResult]]:
        token_ids = self.tokenizer.encode_batch(queries)
        token_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            vectors = self.encoder(token_ids).cpu().numpy()

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
        if not results:
            return "reject"
        top_score = results[0].score
        if top_score >= auto_threshold:
            return "auto"
        elif top_score >= review_threshold:
            return "review"
        else:
            return "reject"

    def rebuild_index(self, products: List[Dict], index_dir: str) -> None:
        names = [p.get("name", "").lower() for p in products]
        vectors = self.encoder.encode_texts(names, self.tokenizer)
        vectors = vectors.astype(np.float32)
        self.index.build(vectors, products)
        self.index.save(index_dir)
