"""
utils/txt_parser.py
-------------------
Parses the supplier TXT price list format.

Header format:  [🍎 Apple][📱 iPhone][iPhone 17 🔥][PRO MAX]
Item format:    [🟢] Name|Price

Example:
    [🍎 Apple][📱 iPhone][iPhone 17 🔥][PRO MAX]
    17 Pro Max 256 Silver (eSim)|101900.00
    🟢 17 PRO MAX 1TB silver (eSim)|137200.00
"""

import re
from typing import Any, Dict, List, Tuple

from utils.keywords import get_keywords
from utils.logger import get_logger

logger = get_logger(__name__)

# Matches lines that consist ONLY of one or more [bracket] groups
_HEADER_RE = re.compile(r"^\s*(\[[^\]]+\])+\s*$")


def _is_header(line: str) -> bool:
    """Return True if the line is a category header ([Brand][Category]...)."""
    return bool(_HEADER_RE.match(line))


def _parse_header(line: str) -> List[str]:
    """Extract all text values from [bracket] groups in a header line."""
    return re.findall(r"\[([^\]]+)\]", line)


def _parse_item(line: str) -> Tuple[str, str, bool]:
    """
    Parse a product item line.

    Returns:
        (name, price, available)
        Empty strings for name/price if the line cannot be parsed.
        available=True when the line started with 🟢.
    """
    line = line.strip()
    available = False

    if line.startswith("🟢"):
        available = True
        line = line[1:].strip()

    if "|" not in line:
        return "", "", False

    parts = line.split("|", 1)
    name = parts[0].strip()
    price = parts[1].strip()

    if not name or not price:
        return "", "", False

    # Validate price is a number
    try:
        float(price)
    except ValueError:
        logger.debug("Skipping line — invalid price '%s': %s", price, line)
        return "", "", False

    return name, price, available


def _match_keywords(name: str, keywords_dict: Dict[str, List[str]]) -> List[str]:
    """
    Find which keywords from keywords.json are present in the product name.
    Returns a deduplicated list of matched keyword strings.
    """
    name_lower = name.lower()
    matched: List[str] = []
    for kws in keywords_dict.values():
        for kw in kws:
            if kw in name_lower and kw not in matched:
                matched.append(kw)
    return matched


def parse_txt(content: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Parse a supplier TXT price list file content.

    Iterates over each line:
      - Header lines (all [brackets]) → update current category context.
      - Item lines (contain '|') → parse name, price, availability.

    Each product is enriched with:
      - brand / category / subcategory / variant from the current header context.
      - keywords: matched keyword strings from data/keywords.json.
      - available: True if the line was prefixed with 🟢.

    Args:
        content: Full text content of the .txt file (UTF-8 decoded).

    Returns:
        Tuple of:
            products      — list of product dicts
            total         — total number of products parsed
            unrecognized  — products whose name matched no keyword in keywords.json
    """
    keywords_dict = get_keywords()
    products: List[Dict[str, Any]] = []
    unrecognized = 0
    current_headers: List[str] = []

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        if _is_header(stripped):
            current_headers = _parse_header(stripped)
            logger.debug("Catalog header: %s", current_headers)
            continue

        name, price, available = _parse_item(stripped)
        if not name or not price:
            continue

        kws = _match_keywords(name, keywords_dict)
        if not kws:
            unrecognized += 1

        product: Dict[str, Any] = {
            "name":        name,
            "price":       price,
            "available":   available,
            "brand":       current_headers[0] if len(current_headers) > 0 else "",
            "category":    current_headers[1] if len(current_headers) > 1 else "",
            "subcategory": current_headers[2] if len(current_headers) > 2 else "",
            "variant":     current_headers[3] if len(current_headers) > 3 else "",
            "keywords":    kws,
        }
        products.append(product)

        logger.debug(
            "Parsed: '%s' | price=%s | available=%s | kw=%s",
            name, price, available, kws,
        )

    logger.info(
        "TXT parse complete: %d products, %d unrecognized.",
        len(products), unrecognized,
    )
    return products, len(products), unrecognized
