"""
neuro/scripts/evaluate.py
--------------------------
Оценка качества обученной модели.

Запуск из корня проекта (папка ai/):
    python -m neuro.scripts.evaluate

Прогоняет тестовые запросы через модель и выводит:
1. Таблицу с примерами матчинга (запрос → результат → score)
2. Метрики: Recall@1, Recall@5, MRR

Тестовые запросы включают:
- Кириллицу ('айфон 13 128 черный')
- Латиницу ('iphone 13 128gb black')
- Опечатки ('iphon 13 blak 128')
- Перестановки ('128 black iphone 13')
"""

import os
import sys

import numpy as np

# Добавляем корень проекта в PATH
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from neuro import config
from neuro.augment.generator import generate_variants
from neuro.dataset.loader import extract_product_names, load_products
from neuro.inference.matcher import ProductMatcher
from neuro.training.metrics import mean_reciprocal_rank, recall_at_k


def main():
    """Главная функция оценки."""
    print("=" * 60)
    print("  ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
    print("=" * 60)

    # ─── 1. Загрузка матчера ──────────────────────────────────────────────
    print("\n[1/3] Загрузка модели и индекса...")
    matcher = ProductMatcher(
        weights_dir=config.WEIGHTS_DIR,
        index_dir=config.INDEX_DIR,
        dimension=config.OUTPUT_DIM,
    )
    print("  Готово.")

    # ─── 2. Тестовые запросы ──────────────────────────────────────────────
    print("\n[2/3] Запуск тестовых запросов...")

    test_queries = [
        "айфон 13 128 черный",
        "iphone 13 128gb black",
        "iphon 13 blak 128",
        "13 айф 128 чёрный",
        "айпад аир 11",
        "ipad air 11",
        "макбук про 14",
        "macbook pro 14",
        "эпл вотч ультра 2",
        "apple watch ultra 2",
        "самсунг с24 256 чёрный",
        "samsung s24 256 black",
        "аирподс про 2",
        "airpods pro 2",
    ]

    print(
        f"\n{'Запрос':<35} {'Результат':<40} {'Score':>6}"
    )
    print("-" * 85)

    for query in test_queries:
        results = matcher.match(query, top_k=1)
        if results:
            top = results[0]
            confidence = matcher.is_confident(results)
            icon = {
                "auto": "✅",
                "review": "⚠️",
                "reject": "❌",
            }[confidence]
            print(
                f"{query:<35} "
                f"{top.product.get('name', '?'):<40} "
                f"{top.score:>5.3f} {icon}"
            )
        else:
            print(
                f"{query:<35} {'НЕ НАЙДЕНО':<40} {'':>6} ❌"
            )

    # ─── 3. Метрики ──────────────────────────────────────────────────────
    print("\n[3/3] Вычисление метрик...")
    products = load_products(config.PRODUCTS_FILE)
    product_names = extract_product_names(products)

    val_texts = []
    val_labels = []
    index_texts = []
    index_labels = []

    for idx, name in enumerate(product_names):
        index_texts.append(name)
        index_labels.append(idx)
        # По 5 вариантов на каждый товар
        for v in generate_variants(name, count=6)[1:]:
            val_texts.append(v)
            val_labels.append(idx)

    val_vectors = matcher.encoder.encode_texts(
        val_texts, matcher.tokenizer
    )
    index_vectors = matcher.encoder.encode_texts(
        index_texts, matcher.tokenizer
    )

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

    print(f"\n{'Метрика':<20} {'Значение':>10}")
    print("-" * 32)
    print(f"{'Recall@1':<20} {r1:>10.3f}")
    print(f"{'Recall@5':<20} {r5:>10.3f}")
    print(f"{'MRR':<20} {mrr:>10.3f}")

    print("\n" + "=" * 60)
    print("  ОЦЕНКА ЗАВЕРШЕНА")
    print("=" * 60)


if __name__ == "__main__":
    main()
