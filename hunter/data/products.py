"""
data/products.py
----------------
Product catalog storage backed by a JSON file.

Each product dict:
    {
        "name":        "17 PRO MAX 1TB silver (eSim)",
        "price":       "137200.00",
        "available":   true,          # true if original line had 🟢 prefix
        "brand":       "🍎 Apple",
        "category":    "📱 iPhone",
        "subcategory": "iPhone 17 🔥",
        "variant":     "PRO MAX",
        "keywords":    ["iphone", "17 pro max", ...]
    }

On every new TXT upload the entire catalog is REPLACED (clear_and_replace).
"""

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional

import config
from utils.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()

# ─── Normalisation helpers ────────────────────────────────────────────────────

# Strips emoji / special Unicode symbols while keeping letters, digits, spaces
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"   # flag sequences (regional indicators)
    "\U0001F300-\U0001F5FF"   # misc symbols & pictographs
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F700-\U0001F7FF"   # alchemical / geometric shapes
    "\U0001F800-\U0001F8FF"   # supplemental arrows
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0001FA00-\U0001FAFF"   # chess / symbols
    "\u2300-\u27BF"           # misc technical / dingbats
    "\u2600-\u26FF"           # misc symbols (includes 🟢 area)
    "]+",
    flags=re.UNICODE,
)


# ─── Russian → English substitution map ──────────────────────────────────────
# Multi-word phrases MUST come before single words (longer match first).
_RU_EN: List[tuple] = [
    # ── multi-word phrases first (longer match wins) ──────────────────────
    ("про макс",       "pro max"),
    ("про мах",        "pro max"),
    ("серебристый",    "silver"),
    ("серебряный",     "silver"),
    ("оранжевый",      "orange"),
    ("фиолетовый",     "purple"),
    ("лавандовый",     "lavender"),
    ("натуральный",    "natural"),
    ("пустынный",      "desert"),
    ("зеленый",        "green"),
    ("зелёный",        "green"),
    ("желтый",         "yellow"),
    ("жёлтый",         "yellow"),    # ── Watch / MacBook / multi-word colors ──────────────────────────────────
    ("джет блэк",      "jet black"),
    ("спейс блэк",     "space black"),
    ("спейс грей",     "space gray"),
    ("спейс серый",    "space gray"),
    ("роз голд",       "rose gold"),
    ("лайт блю",       "light blue"),
    ("айс блю",        "ice blue"),
    # ── device names ─────────────────────────────────────────────────────────────
    ("айфон",          "iphone"),
    ("айпад",          "ipad"),
    ("самсунг",        "samsung"),
    ("сяомей",         "xiaomi"),
    ("сяоме",          "xiaomi"),
    ("сяоми",          "xiaomi"),
    ("часы",            "watch"),
    ("вотч",            "watch"),
    ("макбук",          "macbook"),
    ("мак",             "mac"),
    ("аирподс",         "airpods"),
    ("aw",              "watch"),   # AW = Apple Watch abbreviation
    ("аэрподс",         "airpods"),
    ("карандаш",        "pencil"),
    ("пенсил",           "pencil"),
    ("мышь",            "mouse"),
    ("мышка",           "mouse"),
    ("галакси",         "galaxy"),
    # ── model variants ───────────────────────────────────────────────────
    ("про",            "pro"),
    ("макс",           "max"),
    ("мах",            "max"),
    ("мини",           "mini"),
    ("эйр",            "air"),
    ("аир",            "air"),
    ("плюс",           "plus"),
    ("ультра",         "ultra"),
    # ── storage shorthands (specific before general) ─────────────────────
    ("256гб",          "256"),
    ("512гб",          "512"),
    ("128гб",          "128"),
    ("64гб",           "64"),
    ("1тб",            "1tb"),
    ("2тб",            "2tb"),
    ("терабайт",       "tb"),
    ("гигабайт",       "gb"),
    ("тб",             "tb"),
    ("гб",             "gb"),
    # ── SIM type (strip before punctuation removal) ──────────────────────
    ("есим",           "esim"),
    ("еsim",           "esim"),
    # ── connectivity ─────────────────────────────────────────────────────
    ("вайфай",         "wifi"),
    ("лте",            "lte"),
    ("сотовый",        "lte"),
    ("cellular",       "lte"),
    ("серебро",        "silver"),
    ("сильвер",        "silver"),
    ("блю",            "blue"),
    ("блу",            "blue"),
    ("синий",          "blue"),
    ("голубой",        "blue"),
    ("оранж",          "orange"),
    ("блэк",           "black"),
    ("блек",           "black"),
    ("черный",         "black"),
    ("чёрный",         "black"),
    ("черная",         "black"),
    ("чёрная",         "black"),
    ("уайт",           "white"),
    ("вайт",           "white"),
    ("белый",          "white"),
    ("белая",          "white"),
    ("бел",            "white"),   # короткая форма: "бел" → "white"
    ("голд",           "gold"),
    ("золотой",        "gold"),
    ("золотая",        "gold"),
    ("золото",         "gold"),
    ("серый",          "gray"),
    ("серая",          "gray"),
    ("розовый",        "pink"),
    ("розовая",        "pink"),
    ("пинк",           "pink"),
    ("красный",        "red"),
    ("красная",        "red"),
    ("синяя",          "blue"),
    ("голубая",        "blue"),
    ("фиолетовая",     "purple"),
    ("зеленая",        "green"),
    ("зелёная",        "green"),
    ("желтая",         "yellow"),
    ("жёлтая",         "yellow"),
    ("сейдж",          "sage"),
    ("шалфей",         "sage"),
    ("лаванда",        "lavender"),
    ("лавандер",       "lavender"),
    ("блу",            "blue"),
    ("оранжевый",      "orange"),
    ("пм",             "pro max"),
    ("джет",            "jet"),
    ("спейс",           "space"),
    ("слейт",           "slate"),
    ("плам",            "plum"),
    ("виолет",          "purple"),
    ("графит",          "graphite"),
    ("графитовый",     "graphite"),
    ("неви",            "navy"),
    ("лайм",            "lime"),
    ("лаймовый",       "lime"),
    ("лилак",           "lilac"),
    ("лиловый",         "lilac"),
    ("амбер",           "amber"),
    # --- новые цвета ---
    ("ультрамарин",    "ultramarine"),
    ("тил",            "teal"),
    ("бирюзовый",     "teal"),
    ("бирюза",        "teal"),
    ("мист",           "mist"),
    ("дип",            "deep"),
    ("тёмно-синий",    "deep blue"),
    ("темно-синий",    "deep blue"),
    ("космический",    "cosmic"),
    ("космик",         "cosmic"),
    ("дезерт",         "desert"),
    ("дезертный",      "desert"),
    ("натуральный",    "natural"),
    ("натурал",        "natural"),
    ("миднайт",        "midnight"),
    ("полночный",      "midnight"),
    ("скай",           "sky"),     # "скай блю" → "sky blue"
    ("старлайт",       "starlight"),
    ("звёздный",       "starlight"),
    ("звездный",       "starlight"),
    ("зеленый",        "green"),
    ("зелёный",        "green"),
    ("желтый",         "yellow"),
    ("жёлтый",         "yellow"),
    # --- Samsung / Z-серия ---
    ("минт",           "mint"),
    ("мятный",         "mint"),
    ("мята",           "mint"),
    ("шэдоу",          "shadow"),
    ("шадоу",          "shadow"),
    ("флип",           "flip"),
    ("фолд",           "fold"),
    ("зет",            "z"),
    ("з",              "z"),
    ("анкер",          "anchor"),
    ("фе",             "fe"),
]

# Compiled list: (regex_pattern, replacement)
_RU_EN_RE: List[tuple] = [
    (re.compile(r"\b" + re.escape(ru) + r"\b"), en)
    for ru, en in _RU_EN
]


def _translate_ru(text: str) -> str:
    """Replace Russian tech terms with their English equivalents."""
    for pattern, replacement in _RU_EN_RE:
        text = pattern.sub(replacement, text)
    return text


def _normalize(text: str) -> str:
    """
    Convert any product query or product name into a canonical form so that
    both sides of the comparison are in the same representation.

    Canonical form rules
    --------------------
    - Lowercase
    - Emoji stripped
    - Parenthetical blocks removed: (eSim), (1SIM), (ESIM) → gone
    - Any digit immediately followed by '+' → '<digit> plus'  (15+, 16+ …)
    - Digit↔letter boundaries split: '17про' → '17 про', '256gb' → '256 gb'
    - '16e'/'16е' kept intact (model name, not unit)
    - Russian tech terms translated to English
    - Standalone 'gb' stripped (unit after storage number): '256 gb' → '256'
    - SIM-type indicators stripped: esim, 1sim, 2sim, 1 sim
    - Remaining punctuation replaced with spaces
    - Whitespace collapsed

    Examples
    --------
    '17 PRO MAX 1TB silver (eSim)'  → '17 pro max 1tb silver'
    '17 про макс 1тб сильвер esim'  → '17 pro max 1tb silver'
    '15пм256блю'                    → '15 pro max 256 blue'
    '16е 128 черный'                → '16e 128 black'
    '16е128gb black'                → '16e 128 black'
    '16+ 256 teal'                  → '16 plus 256 teal'
    """
    text = _EMOJI_RE.sub("", text.lower())

    # Collapse double vowels — fixes typos in price lists: miidnight→midnight, starliight→starlight
    # But protect known words that legitimately have double vowels
    _DOUBLE_VOWEL_PROTECT = {'green', 'deep', 'teal', 'ieee', 'see', 'free', 'tree', 'feel', 'seed', 'leen', 'neen', 'been', 'teen', 'beer', 'deer', 'steer', 'steel', 'sleep', 'sheep', 'speed', 'need', 'feed', 'reef', 'keen', 'keep'}
    words_before = text.split()
    text = re.sub(r'([aeiou])\1+', r'\1', text)
    # Restore words that were incorrectly collapsed
    words_after = text.split()
    if len(words_before) == len(words_after):
        for i, (wb, wa) in enumerate(zip(words_before, words_after)):
            if wb != wa and wb in _DOUBLE_VOWEL_PROTECT:
                words_after[i] = wb
        text = ' '.join(words_after)
    # Restore 'book' after collapse ('bok' → 'book')
    text = re.sub(r'\bbok\b', 'book', text)

    # Remove parenthesis CHARS only — keep the content so specs inside like
    # (M2/16Gb/256Gb) or (M5,2TB,Wi-Fi) survive and feed storage/chip guards.
    # SIM markers like (eSim) are handled: 'esim'/'1sim' are stripped below.
    text = re.sub(r'[()]', ' ', text)

    # N+ → N plus  (14+, 15+, 16+, any model)
    text = re.sub(r'(\d)\+', r'\1 plus', text)

    # Apple Watch series alias: "series 11" -> "s11".
    # Helps queries like "Series 11, 46mm" map to S11 products.
    text = re.sub(r'\bseries\s*(\d{1,2})\b', r's\1', text)

    # Watch strap sizes: canonicalize S/M and M/L before punctuation strip.
    # Handles Latin + Cyrillic forms: s/m, с/м, m/l, м/л -> sm/ml.
    text = re.sub(r'\b[sс]\s*[/\\]\s*[mм]\b', 'sm', text)
    text = re.sub(r'\b[mм]\s*[/\\]\s*[lл]\b', 'ml', text)

    # Wi-Fi variants → 'wifi' before punctuation strip
    # handles: wi-fi  wi_fi  wifi  вай-фай  вайфай  вай фай
    text = re.sub(r'wi[-_\s]?fi', 'wifi', text)
    text = re.sub(r'вай[-_]?фай', 'wifi', text)
    text = re.sub(r'\bwf\b', 'wifi', text)   # 'wf' abbreviation for wifi

    # UL (Apple Watch Ultra abbreviation in price lists) → 'ultra'
    text = re.sub(r'\bul\b', 'ultra', text)

    # Samsung series prefixes: Cyrillic а/с before digits → Latin a/s
    # e.g. 'а36' → 'a36', 'с24' → 's24'
    text = re.sub(r'\bа(\d)', r'a\1', text)
    text = re.sub(r'\bс(\d)', r's\1', text)

    # AR shorthand for Air: 'ar13' → 'air 13', 'ar15' → 'air 15'
    text = re.sub(r'\bar\s*(\d{2})\b', r'air \1', text)

    # Cyrillic chip names: 'м4' → 'm4' before digit split breaks them
    text = re.sub(r'м([2-9])', r'm\1', text)

    # Split digit↔letter boundaries
    text = re.sub(r'(\d)([\u0400-\u04FFa-z])', r'\1 \2', text)
    text = re.sub(r'([\u0400-\u04FFa-z])(\d)', r'\1 \2', text)

    # Re-join '16 e' / '16 е' back into '16e' — it's a model name, not a unit
    text = re.sub(r'\b16\s+[еe]\b', '16e', text)

    # Re-join chip names split by digit↔letter: 'm 4' → 'm4', 'm 5' → 'm5'
    text = re.sub(r'\bm\s+([2-9])\b', r'm\1', text)
    # Re-join A-series iPad chip names: 'a 16' → 'a16', 'a 17' → 'a17'
    # Without this — 'A16' in product name splits to 'a 16', and '16' becomes a standalone
    # word that falsely matches 'iPhone 16' queries.
    text = re.sub(r'\ba\s+(1[0-9])\b', r'a\1', text)

    # Russian → English
    text = _translate_ru(text)

    # Canonicalize English synonyms / split compound color words
    text = re.sub(r'\bviolet\b', 'purple', text)          # Samsung 'Violet' → 'purple'
    text = re.sub(r'\blightgray\b', 'light gray', text)   # Samsung 'Lightgray' → 'light gray'
    text = re.sub(r'\bjetblack\b', 'jet black', text)     # Samsung/Watch 'Jetblack'
    text = re.sub(r'\biceblue\b', 'ice blue', text)       # Samsung 'Iceblue'
    text = re.sub(r'\bicyblue\b', 'ice blue', text)       # Samsung 'IcyBlue'
    text = re.sub(r'\bwhitesilver\b', 'white silver', text)    # Samsung 'Whitesilver'
    text = re.sub(r'\bsilvershadow\b', 'silver shadow', text)  # Samsung 'SilverShadow'
    text = re.sub(r'\bblueshadow\b', 'blue shadow', text)     # Samsung 'BlueShadow'
    text = re.sub(r'\bgrey\b', 'gray', text)                  # British grey → gray
    text = re.sub(r'\bsilverblue\b', 'silver blue', text)     # Samsung 'SilverBlue'
    # Watch strap type names -> canonical short codes used in catalog lines.
    text = re.sub(r'\balpine\b', 'al', text)
    text = re.sub(r'\btrail\b', 'tl', text)

    # Strip 'gb' as a standalone unit word ('256 gb' → '256', '128 gb' → '128')
    # Merge digit + space + tb back: '1 tb' → '1tb', '2 tb' → '2tb'
    text = re.sub(r'\bgb\b', '', text)
    text = re.sub(r'\b(\d)\s+tb\b', r'\1tb', text)

    # Strip SIM-type indicators — not part of product identity
    text = re.sub(r'\b(esim|1\s*sim|2\s*sim)\b', '', text)

    # Strip connector type descriptors — not product identifiers
    text = re.sub(r'\btype[-\s]?c\b', '', text)
    text = re.sub(r'\busb[-\s]?c\b', '', text)
    # Normalise 'Pac' → 'Pack' (e.g. "AirTag 4 Pac" → "AirTag 4 Pack")
    text = re.sub(r'\bpac\b', 'pack', text)

    # Re-join Apple SKU patterns split by digit↔letter: 'mw 0 y 3' → 'mw0y3', 'mc 6 j 4' → 'mc6j4'
    # Pattern: (mc|mw) followed by space-separated single chars/digits up to 4 chars total
    text = re.sub(r'\b(m[cw])\s+(\S)\s+(\S)\s+(\S)\b', r'\1\2\3\4', text)
    text = re.sub(r'\b(m[cw])\s+(\S)\s+(\S)\b', r'\1\2\3', text)
    text = re.sub(r'\b(m[cw])\s+(\d+)\b', r'\1\2', text)  # mw 123 → mw123

    # Strip 'mm' unit (e.g. '46mm', '49mm' → '46', '49')
    text = re.sub(r'(\d)\s*mm\b', r'\1', text)

    # Strip remaining punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ─── Persistence ──────────────────────────────────────────────────────────────

def _ensure_file() -> None:
    dirpath = os.path.dirname(config.PRODUCTS_FILE)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    if not os.path.exists(config.PRODUCTS_FILE):
        with open(config.PRODUCTS_FILE, "w", encoding="utf-8") as fh:
            json.dump([], fh)


def load_products() -> List[Dict[str, Any]]:
    """Load and return the full product list from disk."""
    _ensure_file()
    with open(config.PRODUCTS_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_products(products: List[Dict[str, Any]]) -> None:
    """Persist products list to disk (thread-safe)."""
    _ensure_file()
    with _lock:
        with open(config.PRODUCTS_FILE, "w", encoding="utf-8") as fh:
            json.dump(products, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %d products to disk.", len(products))


def clear_and_replace(new_products: List[Dict[str, Any]]) -> None:
    """
    Delete ALL existing products and replace with new_products.
    Called every time a new TXT price list is uploaded.
    """
    save_products(new_products)
    logger.info("Product catalog replaced: %d items.", len(new_products))


def validate_catalog_coverage() -> List[str]:
    """
    Check which products in the catalog are NOT reachable by any alias.

    For each product, normalises its name and runs _resolve_via_aliases —
    the exact same path a real buyer message takes. If it returns None,
    no buyer will ever be able to find this product via the bot.

    Returns a list of product names that have no alias coverage.
    Called after price list upload so the admin can be warned.
    """
    products = load_products()

    uncovered: List[str] = []
    for product in products:
        norm_name = _normalize(product["name"])
        resolved = _resolve_via_aliases(norm_name)
        if resolved is None:
            uncovered.append(product["name"])

    if uncovered:
        logger.warning(
            "Catalog coverage check: %d product(s) have no alias coverage:\n%s",
            len(uncovered),
            "\n".join(f"  - {n}" for n in uncovered),
        )
    else:
        logger.info("Catalog coverage check: all %d products covered.", len(products))

    return uncovered


# ─── Search ───────────────────────────────────────────────────────────────────

# ─── Alias cache (invalidated when keywords.json is modified) ────────────────
_aliases_cache: Optional[Dict[str, List[str]]] = None
_aliases_mtime: float = 0.0
# Pre-normalised form cache: {canonical: (norm_canonical, [norm_form, ...])}
_norm_aliases_cache: Optional[Dict[str, tuple]] = None


def _load_aliases() -> Dict[str, List[str]]:
    """
    Load product_aliases from keywords.json with mtime-based cache.

    The dict is rebuilt from disk only when the file has been modified
    since the last load — otherwise the in-memory cache is returned.
    This gives x10-x100 speedup under load compared to reading on every query.

    Returns dict: {canonical_query: [variant1, variant2, ...]}
    Skips the "_comment" key.
    """
    global _aliases_cache, _aliases_mtime
    import config as _config
    try:
        mtime = os.path.getmtime(_config.KEYWORDS_FILE)
        if _aliases_cache is not None and mtime == _aliases_mtime:
            return _aliases_cache
        with open(_config.KEYWORDS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        aliases = data.get("product_aliases", {})
        _aliases_cache = {k: v for k, v in aliases.items() if k != "_comment" and isinstance(v, list)}
        # Pre-normalise all forms once so _resolve_via_aliases never has to
        global _norm_aliases_cache
        _norm_aliases_cache = {
            canonical: (
                _normalize(canonical),
                [_normalize(v) for v in variants if isinstance(v, str)]
            )
            for canonical, variants in _aliases_cache.items()
        }
        _aliases_mtime = mtime
        logger.debug("Aliases cache refreshed: %d entries.", len(_aliases_cache))
        return _aliases_cache
    except Exception:
        return _aliases_cache or {}


def _resolve_via_aliases(norm_query: str) -> Optional[str]:
    """
    Check if the normalised query matches any alias variant.

    Returns the canonical key (to use as enhanced search query) if found,
    or None if no alias matched.

    Scoring: exact match > substring match; within each tier, longer form wins.
    This prevents 'z fold 7 256 blue' from matching 'z fold 7 256 blue shadow'.
    """
    _load_aliases()  # ensure cache is fresh
    if not _norm_aliases_cache:
        return None
    best_key: Optional[str] = None
    best_score: tuple = (0, 0)  # (exact_flag, length)

    query_words_set = set(norm_query.split())

    for canonical, (norm_canonical, norm_variants) in _norm_aliases_cache.items():
        all_forms = [norm_canonical] + norm_variants
        for form in all_forms:
            if not form:
                continue
            if form == norm_query:
                score: tuple = (2, len(form))   # exact — highest priority
            elif form in norm_query or norm_query in form:
                score = (1, len(form))           # substring
            else:
                # Word-overlap: all words of alias form appear in query
                # Handles long product descriptions (e.g. full Apple Watch names)
                form_words = form.split()
                if len(form_words) >= 3 and all(w in query_words_set for w in form_words):
                    score = (0, len(form))
                else:
                    continue
            if score > best_score:
                best_score = score
                best_key = canonical
    return best_key


# Known iPhone model numbers only (NOT screen sizes, NOT RAM amounts).
# 11/13/15 skipped — they are also MacBook/iPad screen sizes and cause false guards.
# 8/16/24/32/48 are RAM sizes, not model numbers.
_MODEL_NUM_RE   = re.compile(r"^(14|17|25|26)$")  # + Samsung S25/S26 series numbers
_STORAGE_RE     = re.compile(r"^\d{3}$|^\d+tb$")   # 128/256/512 or 1tb/2tb

# ─── SIM-type detection ───────────────────────────────────────────────────────
# Works on RAW (un-normalised) text so we read parenthetical SIM info from
# product names, e.g. "(eSim)" or "(1Sim+eSim)", and plain tokens in queries.
_SIM_ESIM_RE = re.compile(r'\besim\b|e[-]sim\b|еsim|е\s*сим', re.IGNORECASE)
# Standalone \bsim\b (без "1", "e" или "-" перед ним) = физ. SIM: "sim + eSIM"
# Дефис добавлен в lookbehind: "e-sim" не триггерит 1sim
_SIM_1SIM_RE = re.compile(r'\b1\s*sim\b|\b1\s*сим\b|(?<![eе\-])\bsim\b', re.IGNORECASE)

# AirPods generation guard: "AirPods Max 2" ≠ "AirPods MAX USB-C" (без поколения)
# Детектируем: airpods (max|pro) + цифра поколения
_AIRPODS_GEN_RE = re.compile(r'\bairpods\s+(max|pro)\s*(\d)', re.IGNORECASE)


def _sim_type_of(text: str) -> Optional[str]:
    """
    Return the SIM type indicated in *text* (original, un-normalised).

    Returns:
        "1sim"  — physical SIM slot present   → matches products with (1Sim+eSim)
        "esim"  — eSim-only requested         → matches products with (eSim)
        None    — no SIM preference expressed

    Priority: если хи ESIM и 1SIM вместе ("sim + eSIM") — физ. SIM выигрывает.
    """
    has_1sim = bool(_SIM_1SIM_RE.search(text))
    has_esim = bool(_SIM_ESIM_RE.search(text))
    if has_1sim:
        return "1sim"   # физ. SIM-слот (включает eSim в (1Sim+eSim) продуктах)
    if has_esim:
        return "esim"  # Только eSim
    return None

# Screen sizes used in product names (MacBook, iPad, Watch).
# A query containing ONLY these as numbers should not trigger model guard.
_SCREEN_SIZES: frozenset = frozenset(["11", "13", "14", "15", "16", "40", "42", "44", "46", "49"])

# Basic single-word colors used in product names.
# Excludes Apple-specific names (starlight, midnight, titanium…) that may have
# spelling variants in price-list data (e.g. "starliight").
_SIMPLE_COLORS: frozenset = frozenset([
    "white", "black", "silver", "gold", "red", "blue", "green",
    "purple", "pink", "yellow", "orange", "gray", "grey", "teal",
    "starlight", "midnight",   # Apple-specific — важно для color guard
])

# Multi-word colors that must be matched as a whole unit.
# "blue" must NOT match "sky blue"; "gray" must NOT match "space gray", etc.
_COMPOUND_COLORS: List[str] = [
    "sky blue",
    "space gray",
    "space black",
    "rose gold",
    "deep purple",
    "midnight green",
    "sierra blue",
    "alpine green",
    "pacific blue",
    "natural titanium",
    "white titanium",
    "black titanium",
    "desert titanium",
    "anchor blue",
    "deep blue",
    "light blue",
    "dark blue",
    "neon green",
    "yellow green",
    "midnight green",
]


# Apple chip generation tokens: m2, m3, m4, m5 …
_CHIP_RE = re.compile(r'^(?:m[2-9]|a1[0-9]|a[2-9])$')

# MacBook RAM sizes in GB — used to block 16GB queries from matching 24GB products etc.
# 16/24/32/48 are RAM; 8/16 overlap with other contexts but 24/32/48 are unambiguous.
# Guard fires only when BOTH query and product have a RAM token.
_RAM_RE = re.compile(r'^(8|16|24|32|48)$')

# Apple Watch specifics --------------------------------------------------------
# Case sizes (mm) and strap metadata tokens used in product names/queries.
_WATCH_CASE_SIZES: frozenset = frozenset(["40", "42", "44", "46", "49"])
_WATCH_STRAP_TYPES: frozenset = frozenset([
    "al", "tl", "ob", "sb",  # common short codes in watch price lists
    "milanese", "loop", "ocean", "band",
])
# Use only canonical strap-size tokens to avoid collision with model tokens
# (e.g. "s" from "s11" was incorrectly treated as strap size "S").
_WATCH_STRAP_SIZES: frozenset = frozenset(["sm", "ml"])
_WATCH_HINT_WORDS: frozenset = frozenset([
    "watch", "series", "s10", "s11", "se2", "se3", "ul"
])
_SAMSUNG_PHONE_HINTS: frozenset = frozenset([
    "samsung", "galaxy", "s24", "s25", "s26", "fold", "flip", "z"
])

# Generic watch words that can appear in user phrases but are often absent
# in catalog names (e.g. "apple watch series 11 ..." vs "S11 ...").
_WATCH_SCORE_STOPWORDS: frozenset = frozenset(["apple", "watch", "series"])

# Color words used by compound-color guard to detect ambiguous partial color queries.
_COMPOUND_COLOR_WORDS: frozenset = frozenset(
    w for c in _COMPOUND_COLORS for w in c.split()
)

# Russian buy-intent words that appear in user queries but never in product names.
# Filtering them out prevents score dilution: '\u043a\u0443\u043f\u043b\u044e macbook air 13 m4 16 512' → 7 words,
# only 5 match (macbook not in product name) → 5/7=0.71 < 0.75.
# After stripping '\u043a\u0443\u043f\u043b\u044e' → 6 words → 5/6=0.83 ✔
_BUY_INTENT_WORDS: frozenset = frozenset([
    "\u043a\u0443\u043f\u043b\u044e", "\u043a\u0443\u043f\u0438\u0442\u044c", "\u043a\u0443\u043f\u0438", "\u043f\u0440\u0435\u0434\u043b\u043e\u0436\u0438\u0442\u0435", "\u043f\u0440\u0435\u0434\u043b\u043e\u0436\u0438",
    "\u0438\u0449\u0443", "\u043d\u0443\u0436\u0435\u043d", "\u043d\u0443\u0436\u043d\u0430", "\u043d\u0443\u0436\u043d\u043e", "\u043a\u0443\u043f\u043b\u0435\u043c", "\u043a\u0443\u043f\u0438\u043c",
    "\u0446\u0435\u043d\u0430", "\u0446\u0435\u043d\u0443", "\u043f\u043e\u0447\u0435\u043c",
])


def _chip_gen(words: List[str]) -> set:
    """Return chip generation tokens (m2, m3, m4 …) present in a word list."""
    return {w for w in words if _CHIP_RE.match(w)}


def _model_nums(words: List[str]) -> set:
    """Return the set of 2-digit model numbers present in a word list."""
    return {w for w in words if _MODEL_NUM_RE.match(w)}


def _storage_tokens(words: List[str]) -> set:
    """Return storage tokens present in a word list (128, 256, 512, 1tb …)."""
    return {w for w in words if _STORAGE_RE.match(w)}


def _ram_tokens(words: List[str]) -> set:
    """Return RAM size tokens present in a word list (8, 16, 24, 32, 48)."""
    return {w for w in words if _RAM_RE.match(w)}


def _watch_case_tokens(words: List[str]) -> set:
    """Return Apple Watch case size tokens (40/42/44/46/49)."""
    return {w for w in words if w in _WATCH_CASE_SIZES}


def _watch_strap_type_tokens(words: List[str]) -> set:
    """Return watch strap type tokens (AL/TL/OB/SB, ocean, milanese...)."""
    return {w for w in words if w in _WATCH_STRAP_TYPES}


def _watch_strap_size_tokens(words: List[str]) -> set:
    """Return watch strap size tokens (sm/ml/s/m/l)."""
    return {w for w in words if w in _WATCH_STRAP_SIZES}


def _is_watch_context(words: set) -> bool:
    """
    Detect Apple Watch context while avoiding Samsung Ultra false positives.

    Rules:
      - Explicit watch markers => watch context.
      - Bare 'ultra' => watch context only if Samsung phone markers are absent.
    """
    if words & _WATCH_HINT_WORDS:
        return True
    # Normalization splits models: s11 -> "s 11", se3 -> "se 3".
    if "s" in words and ({"10", "11"} & words):
        return True
    if "se" in words and ({"2", "3"} & words):
        return True
    if "ultra" in words and not (words & _SAMSUNG_PHONE_HINTS):
        return True
    return False


def _compound_color_of(norm_name: str) -> Optional[str]:
    """Return the first compound color found in a normalised product name, or None."""
    for color in _COMPOUND_COLORS:
        if color in norm_name:
            return color
    return None


def find_product(text: str) -> Optional[Dict[str, Any]]:
    """
    Find the best-matching product for a user's chat message.

    Algorithm:
        1. Normalise both the query and each product name
           (strip emoji, lowercase, collapse whitespace).
        2. If the entire normalised query is a substring of the product name
           → score = len(query) / len(product_name)  (longer query = higher score).
        3. Otherwise score by fraction of query words found in the product name.
        4. Require score >= 0.75 AND model number from the query must be present
           in the product name (prevents cross-brand/cross-model false matches,
           e.g. "17 Air 256 Black" matching "iPad AIR 13 M2 256 Wi-Fi black").

    Returns:
        Best-matching product dict, or None if no confident match found.
    """
    products = load_products()
    if not products:
        return None

    # Extract SIM preference from the ORIGINAL text (before normalization strips it).
    # "(eSim)" / "(1Sim+eSim)" are in product names; "esim" / "1sim" are in queries.
    query_sim = _sim_type_of(text)

    norm_query = _normalize(text)
    orig_norm_query = norm_query   # keep original for guard extraction

    # ── Step 1: alias lookup (exact user-defined variants) ──────────────────
    alias_query = _resolve_via_aliases(norm_query)
    if alias_query:
        logger.debug("Alias match: '%s' → '%s'", norm_query, alias_query)
        norm_query = alias_query   # use the canonical form for DB search

    query_words = norm_query.split()
    # Guards use the ORIGINAL user query words (pre-alias), not the alias substitution.
    # Alias may drop color/storage words (e.g. "17 pro 512 deep blue" → "17 pro 512 orange"),
    # which would cause guards to miss the user's intent.
    orig_words = orig_norm_query.split()

    # Single-word ORIGINAL queries are always too ambiguous — even if alias expands them
    # to a multi-word string (e.g. "midnight" → "air 13 m4 24 512 midnight"),
    # the guard system relies on orig_words for color/storage context.
    # Without the original context a single word like "midnight" or "ipad" matches
    # the wrong product (no color guard fires because orig_words has no color).
    # EXCEPTION: if alias resolved to an EXACT match (score=2), the single word
    # is an unambiguous product code (e.g. "mc6t4", "mdvn4") — skip the guard.
    if len(orig_words) < 2 and not alias_query:
        return None

    # Also reject if alias expansion still leaves less than 2 score words
    if len(query_words) < 2:
        return None

    # For scoring only: strip Russian buy-intent words (\u043a\u0443\u043f\u043b\u044e, \u043f\u0440\u0435\u0434\u043b\u043e\u0436\u0438\u0442\u0435 \u0435\u0442\u0446.) that dilute overlap score
    # These words never appear in product names, so they only hurt the score.
    score_words = [w for w in query_words if w not in _BUY_INTENT_WORDS]
    if len(score_words) < 2:
        return None

    # 2-digit model numbers in the query (e.g. {17}, {16}, {15})
    query_model   = _model_nums(orig_words)
    # Storage tokens in the query (e.g. {256}, {512}, {1tb})
    query_storage = _storage_tokens(orig_words)
    # Chip generation in the query (e.g. {m4}, {m2}).
    # Если в оригинале нет чипа, берём из alias-expanded query_words (напр. "sky blue" → alias содержит m4).
    query_chip    = _chip_gen(orig_words) or _chip_gen(query_words)
    # RAM tokens in the query (e.g. {16}, {24}, {32}) — MacBook only
    query_ram     = _ram_tokens(orig_words)
    # Watch guards can use alias-expanded words when raw line is too short.
    query_watch_case = _watch_case_tokens(orig_words) or _watch_case_tokens(query_words)
    query_watch_strap_type = _watch_strap_type_tokens(orig_words) or _watch_strap_type_tokens(query_words)
    query_watch_strap_size = _watch_strap_size_tokens(orig_words) or _watch_strap_size_tokens(query_words)
    orig_words_set = set(orig_words)
    query_words_set = set(query_words)
    query_is_watch = _is_watch_context(orig_words_set) or _is_watch_context(query_words_set)

    # Watch queries are frequently verbose ("apple watch ...").
    # Remove generic words from scoring only; guards still use original words.
    if query_is_watch:
        score_words = [w for w in score_words if w not in _WATCH_SCORE_STOPWORDS]
        if len(score_words) < 2:
            return None

    best: Optional[Dict[str, Any]] = None
    best_score = 0.0

    for product in products:
        norm_name = _normalize(product["name"])
        name_words = set(norm_name.split())

        # If query is clearly about Apple Watch, block non-watch products early.
        if query_is_watch and not _is_watch_context(name_words):
            continue

        # Model-number guard: if the query specifies a model (e.g. "17"),
        # the product name must contain the same number.
        # This stops "17 Air 256 Black" from matching "iPad AIR 13 M2 256 black".
        if query_model and not (query_model & name_words):
            continue

        # Storage guard: "17 pro 256 silver" must NOT match "17 pro max 1tb silver"
        # Also check alias-expanded query_words so mw123→"air 13 m4 16 256 midnight"
        # blocks the 512 product even though orig_words=["mw123"] has no storage token.
        effective_storage = query_storage or _storage_tokens(query_words)
        if effective_storage and not (effective_storage & name_words):
            continue

        # Chip generation guard: m2/m3/m4/m5/a16/a18 — если в запросе указан чип,
        # продукт должен иметь тот же чип. Иначе M2 будет матчить M4.
        # name_chip пустой => продукт вообще не имеет чипа => тоже блокируем, если запрос
        # явно указал чип (чтобы «iPad Air 11 M4» не матчил «iPad 11 A16»).
        name_chip = _chip_gen(name_words)
        if query_chip and name_chip and query_chip != name_chip:
            continue
        # Если у продукта нет чипа в названии, но запрос его содержит — блокируем.
        if query_chip and not name_chip and not _is_watch_context(name_words):
            continue

        # RAM guard: 16GB ≠ 24GB ≠ 32GB — только для MacBook (продукты без RAM в
        # названии пропускаются, т.к. name_ram будет пустым).
        name_ram = _ram_tokens(name_words)
        if query_ram and name_ram and query_ram != name_ram:
            continue

        # LTE/Cellular guard: если пользователь запросил LTE/Cellular, не отвечать
        # WiFi-продуктами. Применяется только к продуктам с 'wifi' в названии
        # (MacBook/Watch без 'wifi' в имени не блокируются).
        if 'lte' in orig_words and 'wifi' in name_words and 'lte' not in name_words:
            continue

        # Variant guard: 'max' must appear in BOTH or NEITHER.
        # Prevents '17 Pro 1TB Silver' matching '17 PRO MAX 1TB silver'.
        if ('max' in name_words) != ('max' in set(orig_words)):
            continue

        # Product-family guard: if query contains 'air', product must also contain 'air'.
        # Prevents «iPad Air 11 M4 128» from matching «iPad 11 A16 128».
        if 'air' in orig_words_set and 'air' not in name_words:
            continue

        # Screen-size guard for MacBook Air: '15' in query must not match '13' product and vice versa.
        # Only applies when query explicitly contains one of these screen sizes.
        # Если в orig_words нет размера экрана — проверяем alias-expanded query_words (напр. MC6T4 → air 13).
        _AIR_SCREENS = {'11', '13', '15'}
        query_screen = _AIR_SCREENS & set(orig_words)
        if not query_screen and alias_query:
            query_screen = _AIR_SCREENS & set(query_words)
        name_screen  = _AIR_SCREENS & name_words
        if query_screen and name_screen and query_screen != name_screen:
            continue

        # Apple Watch guard: case size (40/42/44/46/49) must match if requested.
        # Prevents e.g. 'S11 42 ...' matching 'S11 46 ...'.
        if query_is_watch and query_watch_case:
            name_watch_case = _watch_case_tokens(list(name_words))
            if not name_watch_case or not (query_watch_case & name_watch_case):
                continue

        # Apple Watch guard: strap type (TL/AL/OB/SB/...) must match if requested.
        # Prevents e.g. '... TL ...' matching '... AL ...'.
        if query_is_watch and query_watch_strap_type:
            name_watch_strap_type = _watch_strap_type_tokens(list(name_words))
            if not name_watch_strap_type or not (query_watch_strap_type & name_watch_strap_type):
                continue

        # Apple Watch guard: strap size (SM/ML/S/M/L) must match if requested.
        # Prevents e.g. '(M/L)' matching '(S/M)'.
        if query_is_watch and query_watch_strap_size:
            name_watch_strap_size = _watch_strap_size_tokens(list(name_words))
            if not name_watch_strap_size or not (query_watch_strap_size & name_watch_strap_size):
                continue

        # Compound-color guard: "blue" must NOT match "sky blue".
        # Apply only when query touches this compound color family.
        # This keeps valid matches like "watch ultra 3 black" for products that
        # also include a secondary compound color (e.g. "anchor blue").
        product_compound_color = _compound_color_of(norm_name)
        if product_compound_color:
            color_words = product_compound_color.split()
            color_word_set = set(color_words)
            query_color_words = {w for w in orig_words if w in _SIMPLE_COLORS or w in _COMPOUND_COLOR_WORDS}
            if query_color_words & color_word_set:
                if not all(w in orig_words for w in color_words):
                    continue

        # Simple color guard: ALL queried colors must be present in the product name.
        # "Ultra 3 Black Neon Green Ocean Band" has colors {black, green};
        # a product with only "black" (no "green") must be blocked.
        query_simple_colors = {w for w in orig_words if w in _SIMPLE_COLORS}
        if query_simple_colors and not query_simple_colors.issubset(name_words):
            continue

        # SIM-type guard: "esim" query → only (eSim) products;
        #                 "1sim" query → only (1Sim+eSim) products.
        # Non-phone products (MacBook, iPad…) have no SIM info → no guard applied.
        product_sim = _sim_type_of(product.get("name", ""))
        if query_sim and product_sim and query_sim != product_sim:
            continue

        # AirPods generation guard: "AirPods Max 2" — если запрошено поколение X,
        # в названии продукта должна стоять та же цифра X.
        # "куплю AirPods Max 2 Blue" ≠ "AirPods MAX USB-C blue" (нет поколения → None)
        ap_gen_m = _AIRPODS_GEN_RE.search(text)
        if ap_gen_m:
            ap_type = ap_gen_m.group(1).lower()   # 'max' или 'pro'
            ap_gen  = ap_gen_m.group(2)            # '2', '3', '4' ...
            norm_pname = _normalize(product.get("name", ""))
            if 'airpods' in norm_pname and ap_type in norm_pname:
                # если цифра поколения ещё не есть в названии — блок.
                if ap_gen not in norm_pname.split():
                    continue

        # Containment check (most reliable)
        # Use word-overlap ratio as score (= 1.0 when all query words present in product).
        # This prevents a product with MORE word overlap from beating a true substring match.
        # E.g. query "air 13 m4 24 512 midnight" is contained in "MC6C4 air 13 m4 24 512 midnight"
        # (score 1.0) and must NOT be beaten by "MW133 air 13 m4 16 512 midnight" (5/6 = 0.83).
        if norm_query in norm_name:
            score = sum(1 for w in score_words if w in name_words) / len(score_words)
            if score > best_score:
                best_score = score
                best = product
            continue

        # Word-overlap scoring
        # Short score_words (<=5): require 100% — prevents SKU cross-matching
        # Long score_words (6+): 75% — allows brand prefixes like 'Apple', 'купить', etc.
        overlap = sum(1 for w in score_words if w in name_words)
        if overlap >= 2:
            score = overlap / len(score_words)
            required = 1.0 if len(score_words) <= 5 else 0.75
            if score >= required and score > best_score:
                best_score = score
                best = product

    return best if best_score >= 0.75 else None


def find_all_products(text: str) -> List[Dict[str, Any]]:
    """
    Split ``text`` by newlines, search each non-empty line for a product match,
    deduplicate by product name, and return the ordered list of found products.

    Used when the incoming message is a multi-line list (e.g. a stock list like
    "17 Pro 256 Deep Blue - 5") — every line is matched independently.

    Returns an empty list if no products are found.
    """
    seen: set = set()
    found: List[Dict[str, Any]] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        product = find_product(line)
        if product and product["name"] not in seen:
            seen.add(product["name"])
            found.append(product)

    return found


# ─── Summary ──────────────────────────────────────────────────────────────────

def get_catalog_summary() -> Dict[str, Any]:
    """
    Return a summary dict grouping product counts by brand and category.

    Returns:
        {"total": int, "brands": {"brand_name": {"category": count, ...}, ...}}
    """
    products = load_products()
    brands: Dict[str, Dict[str, int]] = {}

    for p in products:
        brand = p.get("brand") or "Unknown"
        category = p.get("category") or "Unknown"
        brands.setdefault(brand, {})
        brands[brand][category] = brands[brand].get(category, 0) + 1

    return {"total": len(products), "brands": brands}
