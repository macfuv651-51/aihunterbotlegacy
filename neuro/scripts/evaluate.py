"""
neuro/scripts/evaluate.py
--------------------------
Оценка качества обученной модели.

Запуск:  python -m neuro.scripts.evaluate
"""

import os
import sys

import numpy as np

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
    print("=" * 60)
    print("  ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
    print("=" * 60)

    print("\n[1/3] Загрузка модели и индекса...")
    matcher = ProductMatcher(
        weights_dir=config.WEIGHTS_DIR,
        index_dir=config.INDEX_DIR,
        dimension=config.OUTPUT_DIM,
    )
    print("  Готово.")

    print("\n[2/3] Запуск тестовых запросов...")

    # Real queries matching actual catalog products
    test_queries = [
        # ── iPhone ──
        "17 про макс 256 силвер есим",
        "17 pro max 1tb blue esim",
        "17 аир 256 блэк",
        "17e 256 white esim",
        "16 про макс 256 блэк",
        "16 128 black",
        "15 128 миднайт",
        "15 pro 128 blue",
        "14 128 midnight",
        "13 128 midnight",
        # ── iPad ──
        "айпад 11 128 блю вайфай",
        "ipad 11 a16 256 wifi pink",
        "айпад эир 11 м4 256",
        "ipad air 13 m4 512 lte",
        "ipad mini 7 512",
        "ipad pro 11 m4 256 silver",
        # ── Apple Watch ──
        "ультра 3 блэк оушен бэнд",
        "ul 2 black black",
        "s11 46 jet black",
        "se3 40 midnight",
        "s10 42 джет блэк",
        # ── MacBook ──
        "air 13 m4 16 256 midnight",
        "аир 15 м4 16 512 скай блю",
        "air 13 m5 24 1tb silver",
        "neo 13 silver",
        "pro 14 gray",
        # ── AirPods ──
        "аирподс 4",
        "airpods pro 2 type c",
        "airpods max usb c blue",
        "airpods pro 3",
        # ── Samsung ──
        "galaxy s26 ultra 16 1tb black",
        "самсунг с25 ультра 12 512",
        "galaxy a36 5g 8 128 lime",
        "galaxy buds 4 pro black",
        "galaxy z fold7 12 256",
        # ── Xiaomi ──
        "сяоми 15 12 256 блэк",
        "note 15 pro 8 256 black",
        "poco x8 pro 8 256 black",
        "poco m7 6 128 black",
        "redmi pad 2 6 128 gray",
        # ── Dyson ──
        "дайсон hd17 суперсоник r джаспер плам с кейсом",
        "dyson v16 ds60 piston animal",
        "dyson hs05 long nickel copper",
        "dyson hs08 amber silk airwrap",
        # ── Other ──
        "jbl flip 7 black",
        "sony wh ch720n black",
        "nintendo switch oled 64 white",
        "мэджик маус юсб с блэк",
        "pixel 10 5g 256 frost",
        "oneplus 15 16 512 ultra violet",
        "ps5 pro digital 2tb",
    ]

    print(f"\n{'Запрос':<42} {'Результат':<45} {'Score':>6}")
    print("-" * 100)

    for query in test_queries:
        results = matcher.match(query, top_k=1)
        if results:
            top = results[0]
            confidence = matcher.is_confident(results)
            icon = {"auto": "V", "review": "?", "reject": "X"}[confidence]
            print(
                f"{query:<42} "
                f"{top.product.get('name', '?'):<45} "
                f"{top.score:>5.3f} {icon}"
            )
        else:
            print(f"{query:<42} {'NOT FOUND':<45} {'':>6} X")

    print("\n[3/3] Вычисление метрик...")
    products = load_products(config.PRODUCTS_FILE)
    product_names = extract_product_names(products)

    val_texts, val_labels = [], []
    index_texts, index_labels = [], []

    for idx, name in enumerate(product_names):
        index_texts.append(name)
        index_labels.append(idx)
        for v in generate_variants(name, count=6)[1:]:
            val_texts.append(v)
            val_labels.append(idx)

    val_vectors = matcher.encoder.encode_texts(val_texts, matcher.tokenizer)
    index_vectors = matcher.encoder.encode_texts(index_texts, matcher.tokenizer)

    val_labels_arr = np.array(val_labels)
    index_labels_arr = np.array(index_labels)

    r1 = recall_at_k(val_vectors, val_labels_arr, index_vectors, index_labels_arr, k=1)
    r5 = recall_at_k(val_vectors, val_labels_arr, index_vectors, index_labels_arr, k=5)
    mrr = mean_reciprocal_rank(val_vectors, val_labels_arr, index_vectors, index_labels_arr)

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
