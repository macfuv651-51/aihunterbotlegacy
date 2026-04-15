"""
neuro/scripts/train.py
----------------------
Точка входа для обучения нейросети матчинга товаров.

Запуск из корня проекта (папка ai/):
    python -m neuro.scripts.train

Этапы:
1. Загрузка products.json
2. Генерация синтетических данных (100 вариантов на товар)
3. Формирование троек (anchor, positive, negative)
4. Создание символьного токенайзера
5. Создание модели (Transformer Encoder)
6. Обучение (Triplet Loss + cosine decay LR)
7. Валидация (Recall@1, Recall@5, MRR) на каждой эпохе
8. Сохранение лучших весов (по Recall@1)

После обучения запустить build_index.py для создания FAISS индекса.
"""

import os
import sys
import time

# Ограничиваем потоки ДО импорта TensorFlow — иначе не работает
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
# Отключаем лишние логи TF (оставляем только ошибки)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

# Добавляем корень проекта в PATH для корректных импортов
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from neuro import config
from neuro.augment.generator import generate_triplets, generate_variants
from neuro.dataset.loader import extract_product_names, load_products
from neuro.dataset.triplet_dataset import create_triplet_dataset
from neuro.model.encoder import ProductEncoder
from neuro.tokenizer.char_tokenizer import CharTokenizer
from neuro.training.trainer import Trainer


def main():
    """Главная функция обучения."""
    print("=" * 60)
    print("  ОБУЧЕНИЕ НЕЙРОСЕТИ МАТЧИНГА ТОВАРОВ")
    print("=" * 60)

    # ─── 1. Загрузка каталога ─────────────────────────────────────────────
    print("\n[1/7] Загрузка каталога товаров...")
    products = load_products(config.PRODUCTS_FILE)
    product_names = extract_product_names(products)
    print(f"  Загружено {len(products)} товаров.")

    # ─── 2. Генерация синтетических данных ────────────────────────────────
    print("\n[2/7] Генерация синтетических данных...")
    start = time.time()
    triplets = generate_triplets(
        products,
        variants_per_product=config.AUGMENT_PER_PRODUCT,
        triplets_per_product=config.TRIPLETS_PER_PRODUCT,
    )
    elapsed = time.time() - start
    print(
        f"  Сгенерировано {len(triplets)} троек "
        f"за {elapsed:.1f}s."
    )

    # ─── 3. Подготовка валидационных данных ────────────────────────────────
    print("\n[3/7] Подготовка валидационных данных...")
    val_texts = []
    val_labels = []
    index_texts = []
    index_labels = []

    for idx, name in enumerate(product_names):
        # Индексные тексты — оригинальные имена товаров
        index_texts.append(name)
        index_labels.append(idx)

        # Валидационные — по 3 сгенерированных варианта на товар
        variants = generate_variants(name, count=4)
        for v in variants[1:]:  # Пропускаем оригинал
            val_texts.append(v)
            val_labels.append(idx)

    print(
        f"  Валидация: {len(val_texts)} запросов "
        f"→ {len(index_texts)} товаров."
    )

    # ─── 4. Создание токенайзера ──────────────────────────────────────────
    print("\n[4/7] Создание токенайзера...")
    all_texts = (
        [t[0] for t in triplets]
        + [t[1] for t in triplets]
        + [t[2] for t in triplets]
        + val_texts
        + index_texts
    )
    tokenizer = CharTokenizer(max_len=config.MAX_SEQ_LEN)
    tokenizer.fit(all_texts)
    print(f"  Словарь: {tokenizer.vocab_size} символов.")

    # ─── 5. Создание модели ───────────────────────────────────────────────
    print("\n[5/7] Создание модели...")
    import tensorflow as tf

    # Ограничиваем потребление памяти — критично для 8 ГБ ОЗУ
    tf.config.threading.set_intra_op_parallelism_threads(
        config.TF_INTRA_THREADS
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        config.TF_INTER_THREADS
    )
    # Ограничиваем GPU memory (на случай если есть видеокарта)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    model = ProductEncoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        ff_dim=config.FF_DIM,
        num_layers=config.NUM_LAYERS,
        output_dim=config.OUTPUT_DIM,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout=config.DROPOUT_RATE,
    )

    # Инициализация весов через один прямой проход
    dummy = tf.zeros((1, config.MAX_SEQ_LEN), dtype=tf.int32)
    model(dummy)

    total_params = model.count_params()
    size_mb = total_params * 4 / 1024 / 1024
    print(f"  Параметров: {total_params:,} ({size_mb:.1f} MB)")

    # ─── 6. Подготовка батчей ─────────────────────────────────────────────
    print("\n[6/7] Подготовка батчей...")
    train_dataset = create_triplet_dataset(
        triplets,
        tokenizer,
        batch_size=config.BATCH_SIZE,
    )
    val_data = (val_texts, val_labels, index_texts, index_labels)

    # ─── 7. Обучение ──────────────────────────────────────────────────────
    print("\n[7/7] Обучение...")
    print("-" * 60)

    trainer = Trainer(
        model=model,
        learning_rate=config.LEARNING_RATE,
        min_lr=config.MIN_LEARNING_RATE,
        margin=config.TRIPLET_MARGIN,
        checkpoint_dir=config.WEIGHTS_DIR,
    )

    history = trainer.train(
        train_dataset=train_dataset,
        val_data=val_data,
        epochs=config.EPOCHS,
        tokenizer=tokenizer,
    )

    # Сохраняем токенайзер рядом с весами
    tokenizer_path = os.path.join(config.WEIGHTS_DIR, "tokenizer.json")
    tokenizer.save(tokenizer_path)

    print("\n" + "=" * 60)
    print("  ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"  Лучший Recall@1: {trainer.best_recall:.3f}")
    print(f"  Веса сохранены в: {config.WEIGHTS_DIR}")
    print("=" * 60)
    print("\nСледующий шаг:")
    print("  python -m neuro.scripts.build_index")


if __name__ == "__main__":
    main()
