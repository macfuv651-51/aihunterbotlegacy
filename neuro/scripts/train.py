"""
neuro/scripts/train.py  (v2 — online triplet generation)
---------------------------------------------------------
Обучение нейросети матчинга товаров на PyTorch.

Пайплайн:
  1. Загрузка 617 товаров из products_full.json
  2. Генерация 5 000 вариантов × 617 = ~3 М строк
  3. Токенизация всех вариантов → numpy-массивы (~300 МБ)
  4. OnlineTripletDataset сэмплирует триплеты на лету
  5. Обучение 30 эпох с ранней остановкой

Запуск:  python -m neuro.scripts.train
"""

import os
import sys
import time

import numpy as np
import torch

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from torch.utils.data import DataLoader

from neuro import config
from neuro.augment.generator import generate_variants
from neuro.dataset.loader import extract_product_names, load_products
from neuro.dataset.triplet_dataset import OnlineTripletDataset, _worker_init_fn
from neuro.model.encoder import ProductEncoder
from neuro.tokenizer.char_tokenizer import CharTokenizer
from neuro.training.trainer import Trainer


# Fixed vocabulary: all characters the model will ever see
_VOCAB_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "0123456789"
    " _"
)


def main():
    print("=" * 60)
    print("  ОБУЧЕНИЕ НЕЙРОСЕТИ МАТЧИНГА ТОВАРОВ v2 (PyTorch)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Устройство: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ─── 1. Load products ────────────────────────────────────────────────
    print("\n[1/8] Загрузка каталога товаров...")
    products = load_products(config.PRODUCTS_FILE)
    product_names = extract_product_names(products)
    print(f"  Загружено {len(products)} товаров.")

    # ─── 2. Tokenizer (fixed vocab, instant) ─────────────────────────────
    print("\n[2/8] Создание токенайзера...")
    tokenizer = CharTokenizer(max_len=config.MAX_SEQ_LEN)
    tokenizer.fit([_VOCAB_CHARS])
    print(f"  Словарь: {tokenizer.vocab_size} символов.")

    # ─── 3. Generate & tokenize variants ─────────────────────────────────
    print(f"\n[3/8] Генерация {config.AUGMENT_PER_PRODUCT} вариантов × "
          f"{len(product_names)} товаров...")
    gen_start = time.time()

    tokenized_variants = {}
    total_variants = 0

    for idx, name in enumerate(product_names):
        variants = generate_variants(name, count=config.AUGMENT_PER_PRODUCT)
        tokenized_variants[idx] = tokenizer.encode_batch(variants)
        total_variants += len(variants)

        if (idx + 1) % 100 == 0 or idx + 1 == len(product_names):
            elapsed = time.time() - gen_start
            print(f"  [{idx+1}/{len(product_names)}] "
                  f"{total_variants:,} вариантов за {elapsed:.1f}s",
                  flush=True)

    # Memory estimate
    mem_bytes = sum(v.nbytes for v in tokenized_variants.values())
    mem_mb = mem_bytes / 1024 / 1024
    print(f"  Итого: {total_variants:,} вариантов, {mem_mb:.0f} MB RAM")

    # ─── 4. Validation data ──────────────────────────────────────────────
    print("\n[4/8] Подготовка валидационных данных...")
    val_texts, val_labels = [], []
    index_texts, index_labels = [], []

    for idx, name in enumerate(product_names):
        index_texts.append(name)
        index_labels.append(idx)
        for v in generate_variants(name, count=4)[1:]:
            val_texts.append(v)
            val_labels.append(idx)

    print(f"  Валидация: {len(val_texts)} запросов → "
          f"{len(index_texts)} товаров.")

    # ─── 5. Model ────────────────────────────────────────────────────────
    print("\n[5/8] Создание модели...")
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
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / 1024 / 1024
    print(f"  Параметров: {total_params:,} ({size_mb:.1f} MB)")

    # ─── 6. Online dataset + DataLoader ──────────────────────────────────
    print(f"\n[6/8] Подготовка DataLoader "
          f"(dataset_size={config.DATASET_SIZE:,})...")
    dataset = OnlineTripletDataset(tokenized_variants, config.DATASET_SIZE)
    train_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    print(f"  Батчей: {len(train_loader)}")

    # ─── 7. Validation tuple ─────────────────────────────────────────────
    val_data = (val_texts, val_labels, index_texts, index_labels)

    # ─── 8. Training ─────────────────────────────────────────────────────
    print(f"\n[7/8] Обучение (до {config.EPOCHS} эпох, "
          f"patience={config.EARLY_STOPPING_PATIENCE})...")
    print("-" * 60)

    trainer = Trainer(
        model=model,
        learning_rate=config.LEARNING_RATE,
        min_lr=config.MIN_LEARNING_RATE,
        margin=config.TRIPLET_MARGIN,
        checkpoint_dir=config.WEIGHTS_DIR,
        device=device,
    )

    history = trainer.train(
        train_loader=train_loader,
        val_data=val_data,
        epochs=config.EPOCHS,
        tokenizer=tokenizer,
        patience=config.EARLY_STOPPING_PATIENCE,
        val_every=config.VAL_EVERY_N_EPOCHS,
    )

    # ─── Save tokenizer ─────────────────────────────────────────────────
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
