"""
Regression tests for product alias resolution.

Test cases are generated DYNAMICALLY from data/keywords.json.
Every alias listed there must resolve to its canonical key.

How to add coverage: add a new alias to keywords.json → test appears automatically.
No need to touch this file.

Run:
    pytest tests/ -v
"""

import json
from pathlib import Path

import pytest

from data.products import _normalize, _resolve_via_aliases

_KEYWORDS_FILE = Path(__file__).parent.parent / "data" / "keywords.json"


def _load_cases():
    with _KEYWORDS_FILE.open(encoding="utf-8") as fh:
        data = json.load(fh)
    aliases = data.get("product_aliases", {})
    cases = []
    for canonical, forms in aliases.items():
        if canonical.startswith("_") or not isinstance(forms, list):
            continue
        # The canonical key itself must resolve to itself
        cases.append((canonical, canonical))
        # Every listed alias must also resolve to this canonical key
        for form in forms:
            if isinstance(form, str):
                cases.append((form, canonical))
    return cases


_CASES = _load_cases()


@pytest.mark.parametrize(
    "form,expected",
    _CASES,
    ids=[f"{c[1]} ← {c[0]}" for c in _CASES],
)
def test_resolve(form: str, expected: str) -> None:
    norm = _normalize(form)
    got = _resolve_via_aliases(norm)
    assert got == expected, (
        f"\n  form:     {form!r}"
        f"\n  norm:     {norm!r}"
        f"\n  got:      {got!r}"
        f"\n  expected: {expected!r}"
    )
