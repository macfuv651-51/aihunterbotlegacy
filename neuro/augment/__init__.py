"""Генерация синтетических обучающих данных: опечатки, транслит, перестановки."""

from neuro.augment.generator import generate_training_pairs, generate_triplets
from neuro.augment.noise import apply_random_noise

__all__ = ["generate_training_pairs", "generate_triplets", "apply_random_noise"]
