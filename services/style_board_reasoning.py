"""Deterministic "why it works" reasoning for Style This boards.

Pure module: no LLM calls, no router imports. Shared by the initial Style
This build (routers/stylist.py) and post-shuffle regeneration
(services/style_board_shuffle_service.py) so both produce the exact same
sentence shape from an anchor item, the board's final items, and its
style_strategy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.style_item_contract import canonical_item_id, canonical_item_role

FALLBACK_STYLING_NOTE = "Built from pieces you already own, anchored on this item."


def _txt(value: Any) -> str:
    return str(value or "").strip()


def _anchor_desc(anchor: Dict[str, Any]) -> str:
    parts = [
        _txt(anchor.get(k))
        for k in ("color_name", "color", "sub_category", "subcategory", "category", "name")
    ]
    seen, words = set(), []
    for p in parts:
        low = p.lower()
        if p and low not in seen:
            seen.add(low)
            words.append(p)
    return (" ".join(words) or "wardrobe item")[:120]


def build_styling_note(
    anchor_item: Optional[Dict[str, Any]],
    final_items: List[Dict[str, Any]],
    style_strategy: Optional[Dict[str, Any]],
    changed_slots: Optional[List[str]] = None,
) -> str:
    """Deterministically describe why the FINAL items work with the anchor.

    Mirrors the sentence shape the initial Style This build has always used:
    "{support} complements {anchor}, keeping the {direction_title} direction
    {reasoning_intent}." Falls back to FALLBACK_STYLING_NOTE when there is no
    style_strategy (matching the pre-existing no-strategy behavior). When
    changed_slots is available, it is authoritative over final item order.
    """
    if not isinstance(style_strategy, dict) or not style_strategy:
        return FALLBACK_STYLING_NOTE

    anchor = anchor_item if isinstance(anchor_item, dict) else {}
    anchor_id = canonical_item_id(anchor) if anchor else ""
    anchor_name = _txt(anchor.get("name") or anchor.get("label")) or _anchor_desc(anchor)

    support_candidates = [
        item for item in (final_items or [])
        if isinstance(item, dict)
        and _txt(item.get("name"))
        and (not anchor_id or canonical_item_id(item) != anchor_id)
        and not item.get("locked")
    ]
    changed_slot_order: List[str] = []
    for value in changed_slots or []:
        slot = _txt(value).lower()
        if slot and slot not in changed_slot_order:
            changed_slot_order.append(slot)

    if changed_slot_order:
        support = next(
            (
                _txt(item.get("name"))
                for slot in changed_slot_order
                for item in support_candidates
                if canonical_item_role(item) == slot
            ),
            "the supporting pieces",
        )
    else:
        support = (
            _txt(support_candidates[0].get("name"))
            if support_candidates
            else "the supporting pieces"
        )

    direction_title = _txt(style_strategy.get("direction_title")) or "this direction"
    intent = (_txt(style_strategy.get("reasoning_intent")) or "intentional").replace(", ", " and ")

    return (
        f"{support} complements {anchor_name}, keeping the {direction_title} "
        f"direction {intent.lower()}."
    )
