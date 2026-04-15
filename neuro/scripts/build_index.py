"""
neuro/scripts/build_index.py
-----------------------------
Построение FAISS индекса из текущего каталога товаров.

Запуск из корня проекта (папка ai/):
    python -m neuro.scripts.build_index

Этапы:
1. Загрузка обученной модели и токенайзера из weights/
2. Загрузка каталога товаров из products.json
3. Кодирование всех товаров в векторы через модель
4. Построение FAISS-индекса и сохранение на диск

Запускать:
- После первого обучения (python -m neuro.scripts.train)
- После каждого обновления products.json (новый прайс-лист)
"""

import os
import sys
import time

import numpy as np

# Добавляем корень проекта в PATH
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from neuro import config
from neuro.dataset.loader import load_products
from neuro.inference.index import FAISSIndex
from neuro.model.encoder import ProductEncoder
from neuro.tokenizer.char_tokenizer import CharTokenizer


def main():
    """Главная функция построения индекса."""
    print("=" * 60)
    print("  ПОСТРОЕНИЕ FAISS ИНДЕКСА")
    print("=" * 60)

    # ─── 1. Загрузка модели ───────────────────────────────────────────────
    print("\n[1/4] Загрузка модели...")
    tokenizer_path = os.path.join(config.WEIGHTS_DIR, "tokenizer.json")
    tokenizer = CharTokenizer.load(tokenizer_path)
    encoder = ProductEncoder.load_all(config.WEIGHTS_DIR)
    print("  Модель загружена.")

    # ─── 2. Загрузка каталога ─────────────────────────────────────────────
    print("\n[2/4] Загрузка каталога...")
    products = load_products(config.PRODUCTS_FILE)
    names = [p.get("name", "").lower() for p in products]
    print(f"  Товаров: {len(products)}")

    # ─── 3. Кодирование ──────────────────────────────────────────────────
    print("\n[3/4] Кодирование товаров в векторы...")
    start = time.time()
    vectors = encoder.encode_texts(names, tokenizer)
    vectors = vectors.astype(np.float32)
    elapsed = time.time() - start
    print(
        f"  Закодировано {len(vectors)} товаров "
        f"за {elapsed:.2f}s."
    )

    # ─── 4. Построение индекса ────────────────────────────────────────────
    print("\n[4/4] Построение FAISS индекса...")
    index = FAISSIndex(dimension=config.OUTPUT_DIM)
    index.build(vectors, products)
    index.save(config.INDEX_DIR)
    print(f"  Индекс сохранён в: {config.INDEX_DIR}")

    print("\n" + "=" * 60)
    print("  ИНДЕКС ГОТОВ")
    print(f"  Размер: {index.size} записей")
    print("=" * 60)
    print("\nМожно тестировать:")
    print("  python -m neuro.scripts.evaluate")


if __name__ == "__main__":
    main()
