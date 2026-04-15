"""Загрузка данных и создание tf.data.Dataset для обучения."""

from neuro.dataset.loader import load_products, extract_product_names
from neuro.dataset.triplet_dataset import create_triplet_dataset

__all__ = ["load_products", "extract_product_names", "create_triplet_dataset"]
