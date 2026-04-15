"""
neuro/augment/generator.py
--------------------------
Генератор обучающих данных из каталога товаров.

Берёт products.json и создаёт тысячи синтетических вариантов
каждого товара с опечатками, перестановками слов и транслитерацией.
Это основной источник данных для обучения нейросети.

Алгоритм:
1. Берём нормализованное имя товара ('iphone 13 128 black')
2. Генерируем N вариантов:
   - Транслитерация → русский ('айфон 13 128 чёрный')
   - Сленг ('13 пм 128')
   - Опечатки ('iphon 13 128 blak')
   - Перестановка слов ('128 black iphone 13')
   - Удаление слов ('iphone 13 128')
   - Комбинации вышеперечисленного
3. Формируем тройки (anchor, positive, negative) для Triplet Loss
"""

import random
from typing import Dict, List, Tuple

from neuro.augment.noise import (
    apply_random_noise,
    drop_random_word,
    shuffle_words,
)
from neuro.dataset.loader import _NORMALIZE as _normalize_name


# ─── Транслитерация English → Russian ────────────────────────────────────────
# Таблица основных технических терминов и их русских аналогов.
# Длинные фразы идут первыми (greedy matching).

_EN_TO_RU: List[Tuple[str, str]] = [
    # Составные цвета (сначала длинные)
    ("space gray", "спейс грей"),
    ("space black", "спейс блэк"),
    ("jet black", "джет блэк"),
    ("rose gold", "роз голд"),
    ("sky blue", "скай блю"),
    ("light blue", "лайт блю"),
    ("deep blue", "дип блю"),
    # Составные модели
    ("pro max", "про макс"),
    # Бренды / устройства
    ("iphone", "айфон"),
    ("ipad", "айпад"),
    ("macbook", "макбук"),
    ("airpods", "аирподс"),
    ("watch", "вотч"),
    ("samsung", "самсунг"),
    ("xiaomi", "сяоми"),
    ("galaxy", "галакси"),
    ("pencil", "пенсил"),
    # Модели
    ("pro", "про"),
    ("max", "макс"),
    ("mini", "мини"),
    ("air", "эйр"),
    ("plus", "плюс"),
    ("ultra", "ультра"),
    # Цвета
    ("silver", "серебро"),
    ("black", "чёрный"),
    ("white", "белый"),
    ("gold", "золотой"),
    ("blue", "синий"),
    ("green", "зелёный"),
    ("red", "красный"),
    ("purple", "фиолетовый"),
    ("pink", "розовый"),
    ("yellow", "жёлтый"),
    ("orange", "оранжевый"),
    ("gray", "серый"),
    ("teal", "бирюзовый"),
    ("midnight", "миднайт"),
    ("starlight", "старлайт"),
    ("desert", "дезерт"),
    ("natural", "натуральный"),
    ("titanium", "титаниум"),
    ("graphite", "графит"),
]


# ─── Сленговые / короткие формы ──────────────────────────────────────────────
# Один термин → несколько возможных сленговых написаний.

_SLANG_FORMS: Dict[str, List[str]] = {
    "pro max": ["пм", "pm", "промакс", "про макс"],
    "pro": ["пр", "про"],
    "iphone": ["айф", "ифон", "айфон"],
    "macbook": ["мак", "макбуг", "макбук"],
    "airpods": ["аирподсы", "подсы", "эирподс"],
    "watch": ["вотч", "часы"],
    "128": ["128gb", "128гб", "128 гб"],
    "256": ["256gb", "256гб", "256 гб"],
    "512": ["512gb", "512гб", "512 гб"],
    "1tb": ["1 тб", "1тб", "1 терабайт"],
}


def _transliterate_to_russian(text: str) -> str:
    """
    Заменить английские технические термины на русские аналоги.

    Порядок замен: сначала длинные фразы, потом короткие,
    чтобы 'pro max' заменился целиком, а не по частям.

    Args:
        text: Строка на английском.

    Returns:
        Строка с русскими аналогами.
    """
    result = text
    for en, ru in _EN_TO_RU:
        if en in result:
            result = result.replace(en, ru)
    return result


def _apply_slang(text: str) -> str:
    """
    Заменить случайный термин на сленговую/короткую форму.

    С вероятностью 30% заменяет каждый найденный термин.
    Не заменяет все подряд — это имитирует реальную речь:
    человек может сказать 'пм' вместо 'pro max', но не все
    слова в запросе будут сленговыми.

    Args:
        text: Исходная строка.

    Returns:
        Строка со сленговой заменой (или без неё).
    """
    result = text
    for original, variants in _SLANG_FORMS.items():
        if original in result and random.random() < 0.3:
            result = result.replace(original, random.choice(variants))
    return result


def generate_variants(
    product_name: str,
    count: int = 100,
) -> List[str]:
    """
    Сгенерировать N вариантов написания одного товара.

    Применяет комбинацию аугментаций с разными вероятностями:
    - 15% — только перестановка слов
    - 15% — транслитерация в русский
    - 15% — транслитерация + опечатки
    - 10% — удаление слова + опечатка
    - 10% — сленг
    - 10% — сленг + перестановка
    - 10% — лёгкие опечатки (1 ошибка)
    - 7%  — средние опечатки (2 ошибки)
    - 8%  — тяжёлые опечатки (3) + перестановка

    Args:
        product_name: Нормализованное имя товара
                      (например, 'iphone 13 128 black').
        count: Количество вариантов для генерации.

    Returns:
        Список строк — различных написаний одного товара.
    """
    variants = [product_name]
    name_lower = product_name.lower()

    for _ in range(count - 1):
        variant = name_lower
        r = random.random()

        if r < 0.15:
            # Только перестановка слов
            variant = shuffle_words(variant)

        elif r < 0.30:
            # Транслитерация в русский
            variant = _transliterate_to_russian(variant)

        elif r < 0.45:
            # Транслитерация + опечатки
            variant = _transliterate_to_russian(variant)
            variant = apply_random_noise(variant, intensity=1)

        elif r < 0.55:
            # Удаление слова + опечатка
            variant = drop_random_word(variant)
            variant = apply_random_noise(variant, intensity=1)

        elif r < 0.65:
            # Сленг
            variant = _apply_slang(variant)

        elif r < 0.75:
            # Сленг + перестановка
            variant = _apply_slang(variant)
            variant = shuffle_words(variant)

        elif r < 0.85:
            # Опечатки (лёгкие, 1 ошибка)
            variant = apply_random_noise(variant, intensity=1)

        elif r < 0.92:
            # Опечатки (средние, 2 ошибки)
            variant = apply_random_noise(variant, intensity=2)

        else:
            # Тяжёлые опечатки (3 ошибки) + перестановка
            variant = apply_random_noise(variant, intensity=3)
            variant = shuffle_words(variant)

        variants.append(variant.strip())

    return variants


def generate_training_pairs(
    products: List[Dict],
    variants_per_product: int = 100,
) -> List[Tuple[str, int]]:
    """
    Сгенерировать обучающие пары (текст, product_index) из каталога.

    Для каждого товара из products.json создаёт N вариантов написания
    и привязывает их к индексу товара (product_id).
    Результат перемешивается.

    Args:
        products: Список словарей из products.json.
        variants_per_product: Количество вариантов на товар.

    Returns:
        Список кортежей (текст_варианта, индекс_товара).
    """
    pairs = []

    for idx, product in enumerate(products):
        name = _normalize_name(product.get("name", ""))
        variants = generate_variants(name, count=variants_per_product)

        for variant in variants:
            pairs.append((variant, idx))

    random.shuffle(pairs)
    return pairs


def generate_triplets(
    products: List[Dict],
    variants_per_product: int = 100,
    triplets_per_product: int = 50,
) -> List[Tuple[str, str, str]]:
    """
    Сгенерировать тройки (anchor, positive, negative) для обучения.

    Anchor и Positive — разные варианты написания ОДНОГО товара.
    Negative — вариант ДРУГОГО товара (случайного).

    Пример:
        anchor   = "айфон 13 128 чёрный"
        positive = "iphon 13 128 black"     (тот же товар)
        negative = "iphone 13 256 blue"     (другой товар)

    Args:
        products: Список словарей из products.json.
        variants_per_product: Количество вариантов для генерации на товар.
        triplets_per_product: Количество троек на товар.

    Returns:
        Список троек (anchor, positive, negative).
    """
    # Генерируем варианты для всех товаров заранее
    all_variants: Dict[int, List[str]] = {}

    for idx, product in enumerate(products):
        raw_name = product.get("name", "")
        name = _normalize_name(raw_name)  # убираем 🇺🇸, al(s), скобки
        all_variants[idx] = generate_variants(
            name, count=variants_per_product
        )

    product_indices = list(all_variants.keys())
    triplets = []

    for idx in product_indices:
        variants = all_variants[idx]
        if len(variants) < 2:
            continue

        for _ in range(triplets_per_product):
            # Anchor и Positive — два случайных варианта одного товара
            anchor, positive = random.sample(variants, 2)

            # Negative — вариант случайного ДРУГОГО товара
            neg_idx = idx
            while neg_idx == idx:
                neg_idx = random.choice(product_indices)
            negative = random.choice(all_variants[neg_idx])

            triplets.append((anchor, positive, negative))

    random.shuffle(triplets)
    return triplets
