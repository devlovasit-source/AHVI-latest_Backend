"""Canonical AHVI Style Orchestrator brief schema.

The Vertex `AHVI_Style_Orchestrator` agent returns loosely-typed styling
intelligence. This module defines the backend-safe canonical shape that the
deterministic style/board engines consume. The agent NEVER generates final
outfits — it only produces this constraint brief; the existing engines and
sanitizers remain the final authority.

Plain dataclasses (no pydantic dependency) — every field has a safe default so
callers can rely on a fully-populated object even from a partial agent payload.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

# Slots the board engine treats as core garments.
ALLOWED_REQUIRED_SLOTS = ("top", "bottom", "footwear", "dress", "one_piece")
# Slots that are nice-to-have / accessory layers, never required.
OPTIONAL_SLOT_VOCAB = ("outerwear", "watch", "belt", "bag", "minimal_jewelry")
# Accessory-ish slots that must be demoted out of required_slots.
ACCESSORY_SLOTS = {"bag", "watch", "belt", "jewelry", "minimal_jewelry"}

WARDROBE_USAGE_VALUES = (
    "owned_first",
    "owned_only",
    "inspiration_only",
    "mixed_owned_and_suggested",
)


@dataclass
class Formality:
    label: str = "casual"
    score: int = 2  # 1..5


@dataclass
class StyleDirection:
    primary: str = ""
    alternates: List[str] = field(default_factory=list)


@dataclass
class PaletteDirection:
    primary: str = ""
    alternates: List[str] = field(default_factory=list)


@dataclass
class CanonicalStyleBrief:
    occasion: str = ""
    sub_intent: str = "outfit_generation"
    formality: Formality = field(default_factory=Formality)
    style_direction: StyleDirection = field(default_factory=StyleDirection)
    wardrobe_usage: str = "owned_first"
    avoid_items: List[str] = field(default_factory=list)
    required_slots: List[str] = field(default_factory=lambda: ["top", "bottom", "footwear"])
    optional_slots: List[str] = field(default_factory=list)
    palette_direction: PaletteDirection = field(default_factory=PaletteDirection)
    accessory_policy: str = ""
    clarification_needed: bool = False
    confidence: float = 0.0
    raw_agent_brief: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "occasion": self.occasion,
            "sub_intent": self.sub_intent,
            "formality": {"label": self.formality.label, "score": self.formality.score},
            "style_direction": {
                "primary": self.style_direction.primary,
                "alternates": list(self.style_direction.alternates),
            },
            "wardrobe_usage": self.wardrobe_usage,
            "avoid_items": list(self.avoid_items),
            "required_slots": list(self.required_slots),
            "optional_slots": list(self.optional_slots),
            "palette_direction": {
                "primary": self.palette_direction.primary,
                "alternates": list(self.palette_direction.alternates),
            },
            "accessory_policy": self.accessory_policy,
            "clarification_needed": self.clarification_needed,
            "confidence": self.confidence,
            "raw_agent_brief": dict(self.raw_agent_brief),
        }
