"""
neuro/augment/generator.py  (v2 — massive augmentation)
-------------------------------------------------------
Генератор обучающих данных: 617 товаров × 5 000 вариантов = ~3 М.

Основные стратегии аугментации:
  • Транслитерация EN→RU (слова и фразы)
  • Сленг / сокращения
  • Формат памяти (256 → 256гб)
  • Удаление слов, перестановка, опечатки
  • Независимая комбинация всех стратегий
"""

import random
from typing import Dict, List, Tuple

from neuro.augment.noise import (
    apply_random_noise,
    drop_random_word,
    shuffle_words,
)
from neuro.dataset.loader import _NORMALIZE as _normalize_name


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTI-WORD PHRASES → Russian alternatives  (applied FIRST, greedy)
# ═══════════════════════════════════════════════════════════════════════════════

_PHRASE_VARIANTS: Dict[str, List[str]] = {
    # ── Apple compound models ──
    "pro max": ["про макс", "промакс"],
    "pro plus": ["про плюс"],

    # ── Compound colours ──
    "space gray": ["спейс грей"],
    "space black": ["спейс блэк"],
    "jet black": ["джет блэк"],
    "rose gold": ["роуз голд", "роз голд"],
    "sky blue": ["скай блю"],
    "light blue": ["лайт блю"],
    "cobalt violet": ["кобальт вайолет"],
    "silver shadow": ["силвер шедоу"],
    "pink gold": ["пинк голд"],
    "ocean band": ["оушен бэнд"],
    "milanese loop": ["миланез луп"],
    "sport band": ["спорт бэнд"],
    "sport loop": ["спорт луп"],
    "mint breeze": ["минт бриз"],
    "nebula noir": ["небула нуар"],
    "astral trail": ["астрал трейл"],
    "black velvet": ["блэк велвет"],
    "green silk": ["грин силк"],

    # ── Apple device pairs ──
    "magic mouse": ["мэджик маус"],
    "usb c": ["юсб с", "тайп си", "type c"],
    "wi fi": ["вайфай"],
    "power adapter": ["зарядка", "блок питания"],

    # ── Dyson ──
    "diffuse for curly": ["для кудрявых"],
    "strait wavy": ["для прямых"],
    "vacuum cleaner": ["пылесос"],
    "hand dryer": ["сушилка для рук"],
    "с кейсом": ["с чехлом", "кейс", "в кейсе"],
    "без кейса": ["без чехла"],
    "наша вилка": ["наша", "рос вилка"],

    # ── Samsung compound ──
    "galaxy s": ["галакси с", "самсунг с"],
    "galaxy a": ["галакси а", "самсунг а"],
    "galaxy z": ["галакси з"],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SINGLE WORD → Russian alternatives
# ═══════════════════════════════════════════════════════════════════════════════

_WORD_VARIANTS: Dict[str, List[str]] = {
    # ── Brands ──
    "iphone": ["айфон", "ифон"],
    "ipad": ["айпад"],
    "macbook": ["макбук"],
    "airpods": ["аирподс", "эирподс"],
    "apple": ["эпл"],
    "samsung": ["самсунг"],
    "galaxy": ["галакси"],
    "xiaomi": ["сяоми", "шаоми"],
    "poco": ["поко"],
    "redmi": ["редми"],
    "dyson": ["дайсон"],
    "sony": ["сони"],
    "jbl": ["джбл", "джибиэль"],
    "nintendo": ["нинтендо"],
    "switch": ["свитч"],
    "honor": ["хонор"],
    "oneplus": ["ванплас", "уанплас"],
    "google": ["гугл"],
    "pixel": ["пиксель"],
    "realme": ["реалми"],
    "playstation": ["плейстейшн"],

    # ── Device types ──
    "watch": ["вотч", "часы"],
    "pencil": ["пенсил"],
    "mouse": ["маус", "мышка"],
    "buds": ["бадс"],
    "pad": ["пад"],
    "book": ["бук"],
    "headphones": ["наушники"],
    "tune": ["тюн"],
    "flip": ["флип"],
    "charge": ["чардж"],
    "станция": ["станция", "станц"],
    "колонка": ["колонка"],
    "стрит": ["стрит"],

    # ── Model variants ──
    "pro": ["про"],
    "max": ["макс"],
    "mini": ["мини"],
    "air": ["эир", "аир"],
    "plus": ["плюс"],
    "ultra": ["ультра"],
    "neo": ["нео"],
    "fold": ["фолд"],
    "lite": ["лайт"],
    "slim": ["слим"],
    "classic": ["классик"],
    "digital": ["диджитал"],
    "oled": ["олед"],
    "supersonic": ["суперсоник"],
    "airwrap": ["эирврэп"],
    "airstrait": ["эирстрейт"],
    "absolute": ["абсолют"],
    "detect": ["детект"],
    "submarine": ["субмарин"],
    "advanced": ["эдвансд"],
    "nural": ["нурал"],
    "pencilvac": ["пенсилвак"],

    # ── Colours ──
    "black": ["блэк", "чёрный", "черный"],
    "white": ["вайт", "белый"],
    "silver": ["силвер", "серебристый"],
    "gold": ["голд", "золотой"],
    "blue": ["блю", "синий", "голубой"],
    "green": ["грин", "зелёный", "зеленый"],
    "red": ["ред", "красный"],
    "purple": ["пёрпл", "фиолетовый"],
    "pink": ["пинк", "розовый"],
    "yellow": ["еллоу", "жёлтый"],
    "orange": ["оранж", "оранжевый"],
    "gray": ["грей", "серый"],
    "grey": ["грей", "серый"],
    "midnight": ["миднайт"],
    "starlight": ["старлайт"],
    "desert": ["дезерт"],
    "natural": ["нэйчурал", "натуральный"],
    "titanium": ["титаниум", "титан"],
    "graphite": ["графит"],
    "teal": ["тил", "бирюзовый"],
    "lavender": ["лавандер"],
    "lavander": ["лавандер"],
    "sage": ["сейдж"],
    "mint": ["минт", "мятный"],
    "navy": ["нэйви"],
    "obsidian": ["обсидиан"],
    "coral": ["корал"],
    "copper": ["коппер", "медный"],
    "nickel": ["никель"],
    "bronze": ["бронза"],
    "plum": ["плам"],
    "ceramic": ["керамик"],
    "amber": ["амбер"],
    "topaz": ["топаз"],
    "onyx": ["оникс"],
    "fuchsia": ["фуксия"],
    "charcoal": ["чаркоал"],
    "slate": ["слейт"],
    "frost": ["фрост"],
    "indigo": ["индиго"],
    "blush": ["блаш"],
    "cobalt": ["кобальт"],
    "violet": ["вайолет"],
    "lime": ["лайм"],
    "citrus": ["цитрус"],
    "fog": ["фог"],
    "jasper": ["джаспер"],
    "crimson": ["кримсон"],
    "pearl": ["перл"],
    "rose": ["роуз"],
    "silk": ["силк"],
    "velvet": ["велвет", "бархат"],
    "denim": ["деним"],
    "squad": ["сквод"],
    "ivory": ["айвори"],
    "neon": ["неон"],

    # ── Connectivity ──
    "wifi": ["вайфай"],
    "lte": ["лте"],
    "esim": ["есим"],
    "nfc": ["нфс"],

    # ── Storage ──
    "1tb": ["1тб", "1 тб"],
    "2tb": ["2тб", "2 тб"],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  MEMORY FORMAT variants (standalone numbers → with suffixes)
# ═══════════════════════════════════════════════════════════════════════════════

_MEMORY_FORMATS: Dict[str, List[str]] = {
    "128": ["128gb", "128гб", "128 гб"],
    "256": ["256gb", "256гб", "256 гб"],
    "512": ["512gb", "512гб", "512 гб"],
    "64":  ["64gb", "64гб"],
    "32":  ["32gb", "32гб"],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SLANG / shortcuts
# ═══════════════════════════════════════════════════════════════════════════════

_SLANG: Dict[str, List[str]] = {
    "pro max": ["пм", "pm"],
    "iphone": ["айф"],
    "macbook": ["мак"],
    "airpods": ["подсы"],
    "samsung": ["самс"],
    "playstation": ["пс"],
    "ultra": ["ульт"],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _transliterate_random(text: str, prob: float = 0.6) -> str:
    """Replace random English words with Russian equivalents."""
    words = text.split()
    result: List[str] = []
    i = 0

    while i < len(words):
        matched = False
        # Try 3-word, then 2-word phrases
        for phrase_len in (3, 2):
            if i + phrase_len <= len(words):
                phrase = " ".join(words[i : i + phrase_len])
                if phrase in _PHRASE_VARIANTS and random.random() < prob:
                    result.append(random.choice(_PHRASE_VARIANTS[phrase]))
                    i += phrase_len
                    matched = True
                    break

        if not matched:
            word = words[i]
            if word in _WORD_VARIANTS and random.random() < prob:
                result.append(random.choice(_WORD_VARIANTS[word]))
            else:
                result.append(word)
            i += 1

    return " ".join(result)


def _apply_slang(text: str) -> str:
    """Replace terms with slang/short forms."""
    result = text
    for original, variants in _SLANG.items():
        if original in result and random.random() < 0.3:
            result = result.replace(original, random.choice(variants))
    return result


def _apply_memory_format(text: str) -> str:
    """Add storage suffixes: '256' → '256гб'."""
    words = text.split()
    result: List[str] = []
    for word in words:
        if word in _MEMORY_FORMATS and random.random() < 0.3:
            result.append(random.choice(_MEMORY_FORMATS[word]))
        else:
            result.append(word)
    return " ".join(result)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_variants(product_name: str, count: int = 5000) -> List[str]:
    """
    Generate *count* augmented variants of a single product name.

    Each variant is produced by independently applying random transforms:
      55% — partial/full transliteration EN→RU
      15% — slang abbreviation
      15% — memory format change
      15% — drop a random word
      20% — shuffle word order
      20% — add typos (1-3 errors)

    These are independent coin-flips, so ~2^6 = 64 possible combos.
    """
    variants = [product_name]
    name_lower = product_name.lower()

    for _ in range(count - 1):
        variant = name_lower

        if random.random() < 0.55:
            variant = _transliterate_random(
                variant, prob=random.uniform(0.3, 1.0)
            )

        if random.random() < 0.15:
            variant = _apply_slang(variant)

        if random.random() < 0.15:
            variant = _apply_memory_format(variant)

        if random.random() < 0.15:
            variant = drop_random_word(variant)

        if random.random() < 0.20:
            variant = shuffle_words(variant)

        if random.random() < 0.20:
            intensity = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            variant = apply_random_noise(variant, intensity=intensity)

        variants.append(variant.strip())

    return variants


# ═══════════════════════════════════════════════════════════════════════════════
#  TRIPLET GENERATOR  (offline / legacy — kept for compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_training_pairs(
    products: List[Dict],
    variants_per_product: int = 1000,
) -> List[Tuple[str, int]]:
    """(text, product_index) pairs from catalog — legacy helper."""
    pairs = []
    for idx, product in enumerate(products):
        name = _normalize_name(product.get("name", ""))
        for variant in generate_variants(name, count=variants_per_product):
            pairs.append((variant, idx))
    random.shuffle(pairs)
    return pairs


def generate_triplets(
    products: List[Dict],
    variants_per_product: int = 1000,
    triplets_per_product: int = 500,
) -> List[Tuple[str, str, str]]:
    """(anchor, positive, negative) triplets — offline, for small catalogs."""
    all_variants: Dict[int, List[str]] = {}
    for idx, product in enumerate(products):
        name = _normalize_name(product.get("name", ""))
        all_variants[idx] = generate_variants(name, count=variants_per_product)

    product_indices = list(all_variants.keys())
    triplets: List[Tuple[str, str, str]] = []

    for idx in product_indices:
        variants = all_variants[idx]
        if len(variants) < 2:
            continue
        for _ in range(triplets_per_product):
            anchor, positive = random.sample(variants, 2)
            neg_idx = idx
            while neg_idx == idx:
                neg_idx = random.choice(product_indices)
            negative = random.choice(all_variants[neg_idx])
            triplets.append((anchor, positive, negative))

    random.shuffle(triplets)
    return triplets
