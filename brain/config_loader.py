"""AHVI Style config loader.

Loads the curated JSON configuration/intelligence files under
``brain/configs/`` and returns COMPACT, style-relevant policy context.

Design rules (per AHVI MVP Final Integration Guide):
- These JSON files are configuration/intelligence layers, not services.
- NEVER inject full rule libraries into Gemini prompts. Some files (the
  core/personality "rule libraries") are large few-shot example dumps and
  may even contain multiple concatenated JSON documents. We parse them
  best-effort and only ever surface a small hand-curated policy slice.
- Fail gracefully: a missing or malformed file never breaks styling.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List

logger = logging.getLogger("ahvi.config_loader")

_CONFIG_ROOT = os.path.join(os.path.dirname(__file__), "configs")

# Relative paths of the style-MVP config set. Only these are integrated.
STYLE_CONFIG_FILES = {
    "decision_rules": "core/decision_rules.json",
    "boundaries": "core/boundaries.json",
    "personalization": "personality/personalization.json",
    "visual_rules": "personality/visual_rules.json",
    "wardrobe_context": "context/wardrobe_context.json",
    "weather_context": "context/weather_context.json",
    "event_context": "context/event_context.json",
    "user_context_schema": "context/user_context_schema.json",
    "style_board": "experiences/style_board.json",
    "visual_response_strategy": "experiences/visual_response_strategy.json",
    "style_dna_schema": "memory/style_dna_schema.json",
    "wardrobe_memory_schema": "memory/wardrobe_memory_schema.json",
    "outfit_history_schema": "memory/outfit_history_schema.json",
}


def _strip_md_fences(raw: str) -> str:
    """Some source configs were exported with stray markdown code-fence lines
    (``` / ```json) embedded inside the JSON. Drop any line that is only a
    fence so the remaining text parses as JSON."""
    if "```" not in raw:
        return raw
    kept = [ln for ln in raw.splitlines() if not ln.strip().startswith("```")]
    return "\n".join(kept)


def _parse_multi_doc(raw: str) -> Any:
    """Parse a file that may contain one clean JSON document OR several
    concatenated documents (the big rule-library dumps). Returns the first
    object if it's a dict, else a list of all decodable top-level values.
    Best-effort: stops at the first undecodable region instead of raising."""
    raw = _strip_md_fences(raw).strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    values: List[Any] = []
    idx, n = 0, len(raw)
    while idx < n:
        while idx < n and raw[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            break  # malformed tail — keep what we have
        values.append(obj)
        idx = end
    if not values:
        return {}
    if len(values) == 1:
        return values[0]
    return values


@lru_cache(maxsize=64)
def load_config(path: str) -> Dict[str, Any]:
    """Load a single config JSON by path relative to brain/configs/.
    Cached. Returns {} on any failure."""
    abs_path = path if os.path.isabs(path) else os.path.join(_CONFIG_ROOT, path)
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            parsed = _parse_multi_doc(fh.read())
        if isinstance(parsed, dict):
            return parsed
        # Multi-doc / list payloads are wrapped so callers always get a dict.
        return {"_documents": parsed}
    except FileNotFoundError:
        logger.warning("ahvi.config.missing path=%s", path)
        return {}
    except Exception as exc:  # noqa: BLE001 - never break styling on config
        logger.warning("ahvi.config.load_failed path=%s err=%s", path, str(exc)[:160])
        return {}


@lru_cache(maxsize=1)
def load_style_configs() -> Dict[str, Dict[str, Any]]:
    """Load all style-MVP configs once, keyed by logical name. Cached."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, rel in STYLE_CONFIG_FILES.items():
        out[name] = load_config(rel)
    return out


def _board_policy() -> Dict[str, Any]:
    board = load_config(STYLE_CONFIG_FILES["style_board"])
    objective = board.get("experience_objective") or {}
    required = board.get("required_context") or {}
    return {
        "primary_goal": objective.get("primary_goal", ""),
        "secondary_goals": (objective.get("secondary_goals") or [])[:5],
        "critical_context": (required.get("critical") or [])[:6],
    }


def _visual_hierarchy() -> List[str]:
    strat = load_config(STYLE_CONFIG_FILES["visual_response_strategy"])
    hierarchy = strat.get("visual_first_hierarchy")
    if isinstance(hierarchy, list):
        return [str(x) for x in hierarchy][:8]
    return ["board", "cards", "text"]


# Compact, hand-curated policy slices keyed by mode. These are intentionally
# tiny so they can be safely injected into a Gemini prompt. We do NOT dump the
# large rule libraries — we summarise the *intent* the MVP needs.
_MODE_POLICY = {
    "style_advice": {
        "priority": "Lead with stylist reasoning. Explain the social strategy "
        "before any clothing. 3 distinct visual directions.",
        "guardrails": [
            "no generic filler (balanced silhouette / color harmony / "
            "elevated aesthetic / perfect for)",
            "different occasions must produce different goal + impression + avoid",
        ],
    },
    "visual_inspiration": {
        "priority": "Cards are the main response. Each direction must differ by "
        "mood, silhouette, palette, or formality.",
        "guardrails": ["no repeated/templated directions", "no recursive prompt echo"],
    },
    "wardrobe_style": {
        "priority": "Build an editorial board from the user's ACTUAL wardrobe "
        "items. Name the hero, support, footwear, accessory roles.",
        "guardrails": [
            "use real wardrobe images, never placeholders when images exist",
            "do not render as generic module cards or a visual-inspiration carousel",
        ],
    },
    "missing_pieces": {
        "priority": "Identify the smallest set of missing pieces that unlock the "
        "most outfits. Justify each with a reason + what it unlocks.",
        "guardrails": ["one clear reason per item", "list real-world unlocks"],
    },
}


def get_style_policy_context(intent: str, occasion: str, mode: str) -> Dict[str, Any]:
    """Return a COMPACT policy context for the prompt. Small by design.

    Never returns the raw rule libraries — only the curated slice the MVP
    needs for the given mode/occasion."""
    safe_mode = str(mode or "style_advice").strip().lower()
    policy = _MODE_POLICY.get(safe_mode, _MODE_POLICY["style_advice"])
    try:
        board = _board_policy()
        hierarchy = _visual_hierarchy()
    except Exception:  # noqa: BLE001
        board, hierarchy = {}, ["board", "cards", "text"]
    selected = {
        "intent": str(intent or "").strip().lower() or "style_advice",
        "occasion": str(occasion or "").strip().lower() or None,
        "mode": safe_mode,
        "mode_priority": policy["priority"],
        "guardrails": policy["guardrails"],
        "experience_goal": board.get("primary_goal", ""),
        "secondary_goals": board.get("secondary_goals", []),
        "visual_first_hierarchy": hierarchy[:4],
    }
    logger.info(
        "AHVI_STYLE_POLICY_SELECTED intent=%s occasion=%s mode=%s goal=%r",
        selected["intent"],
        selected["occasion"],
        selected["mode"],
        (selected["experience_goal"] or "")[:60],
    )
    return selected
