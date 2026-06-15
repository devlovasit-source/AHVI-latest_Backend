"""Deterministic color-clash logic for the post-LLM visual style guard.

No machine-readable clash map existed in the backend before this:
``color_harmony_bank.json`` is feedback prose and ``ahvi_color_pair_logic_v1.json``
is gendered routing prose. ``_asset_score`` only *rewards* matches. This module
is the small, pure clash map (mirroring the proven Flutter ``PairingEngine``)
that the guard uses to *drop* clashing supporting pieces.

Rules (kept intentionally conservative — over-stripping is worse than a near miss
because the guard's color value scales with color-metadata coverage):
- Neutrals pair with anything.
- A small set of explicitly known-bad pairs always clash.
- For two non-neutral colors that both have harmony data, they clash when
  neither lists the other as compatible.
- Unknown / undefined non-neutral pairs do NOT clash (degrade safe).
- navy/black is special-cased at the palette level (``palette_clashes``):
  only flagged when both are dominant and there is no neutral buffer.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

NEUTRALS: Set[str] = {
    "white",
    "offwhite",
    "off-white",
    "black",
    "grey",
    "gray",
    "beige",
    "cream",
    "ivory",
    "navy",
    "tan",
    "khaki",
    "brown",
    "denim",
    "stone",
    "charcoal",
}

# color -> set of compatible non-neutral partners.
HARMONY = {
    "burgundy": {"cream", "beige", "navy", "grey", "gray", "tan", "white", "olive"},
    "maroon": {"cream", "beige", "navy", "grey", "gray", "tan", "white", "olive"},
    "green": {"white", "beige", "tan", "navy", "black", "cream", "brown", "burgundy"},
    "olive": {"white", "beige", "tan", "navy", "black", "cream", "brown", "burgundy", "rust"},
    "red": {"black", "white", "navy", "denim", "grey", "gray", "cream"},
    "orange": {"navy", "blue", "white", "cream", "brown", "tan"},
    "yellow": {"navy", "blue", "white", "grey", "gray", "denim"},
    "purple": {"grey", "gray", "white", "black", "cream", "silver"},
    "pink": {"navy", "white", "grey", "gray", "cream", "denim", "blue"},
    "blue": {"white", "cream", "tan", "beige", "orange", "grey", "gray"},
    "teal": {"white", "cream", "tan", "beige", "grey", "gray", "rust"},
    "rust": {"cream", "beige", "tan", "olive", "navy", "teal", "white"},
}

# Explicit known-bad pairs (fast, unconditional reject).
CLASH: Set[frozenset] = {
    frozenset({"burgundy", "green"}),
    frozenset({"maroon", "green"}),
    frozenset({"red", "green"}),
    frozenset({"orange", "purple"}),
    frozenset({"yellow", "purple"}),
    frozenset({"pink", "orange"}),
    frozenset({"red", "pink"}),
    frozenset({"brown", "purple"}),
}


def _norm(value: object) -> str:
    return str(value or "").strip().lower()


def colors_clash(a: object, b: object) -> bool:
    """True when two named colors visibly clash.

    Neutrals never clash. Equal/empty never clash. Explicit CLASH pairs always
    clash. Otherwise two colors that both carry harmony data clash only when
    neither lists the other. Unknown non-neutral pairs degrade safe (no clash).
    navy/black is handled at the palette level, not here.
    """
    a, b = _norm(a), _norm(b)
    if not a or not b or a == b:
        return False
    pair = frozenset({a, b})
    if pair in CLASH:
        return True
    if a in NEUTRALS or b in NEUTRALS:
        return False
    ha: Optional[Set[str]] = HARMONY.get(a)
    hb: Optional[Set[str]] = HARMONY.get(b)
    # Both have harmony data: clash only when neither accepts the other.
    if ha is not None and hb is not None:
        return b not in ha and a not in hb
    # One side has harmony data: clash if it explicitly excludes the other.
    if ha is not None:
        return b not in ha
    if hb is not None:
        return a not in hb
    # Neither defined -> unknown pair, degrade safe.
    return False


def palette_clashes(colors: List[object]) -> List[Tuple[str, str]]:
    """Return every clashing pair within a palette.

    Includes the navy/black special case: flagged only when navy and black are
    the dominant colors (no neutral buffer alongside them in the palette).
    """
    cs = [_norm(c) for c in (colors or []) if _norm(c)]
    bad: List[Tuple[str, str]] = []
    for i, a in enumerate(cs):
        for b in cs[i + 1:]:
            if colors_clash(a, b):
                bad.append((a, b))

    # navy/black dominant-without-buffer special case.
    if "navy" in cs and "black" in cs:
        buffer = {c for c in cs if c in NEUTRALS and c not in {"navy", "black"}}
        if not buffer:
            bad.append(("navy", "black"))
    return bad
