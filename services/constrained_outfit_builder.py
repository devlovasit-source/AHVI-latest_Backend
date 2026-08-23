"""Constrained outfit generation with hard fixed-item (lock) guarantees.

Core idea: candidate pools are constrained BEFORE combination.  A fixed slot's
pool contains exactly the fixed item; replaceable slots draw only from
eligible items of that role, filtered by source policy, exclusions and
occasion safety.  The fixed-item invariant
(style_item_contract.assert_fixed_items_preserved) runs on every candidate
outfit and again on the final selection - a fixed item can never be dropped,
swapped or silently replaced.  On any impossibility a TYPED failure dict is
returned; never an unrelated fallback outfit.

Deliberately lightweight: no brain.outfit_pipeline.get_daily_outfits call
(that path is anchor-hostile: closed-loop slot swaps, tops/dress-only masters,
canonicalization drops).  Scoring is a simple deterministic mix of color
harmony, occasion tags and preference tokens with stable-hash rotation for
shuffle variety.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from services.style_board_image_readiness import (
    is_board_renderable,
    prepare_board_item,
)
from services.style_item_contract import (
    FixedItemLostError,
    VALID_SOURCES,
    assert_fixed_items_preserved,
    canonical_item_id,
    canonical_item_role,
    canonical_item_source,
    normalize_style_item,
)
from services.professional_safety import (
    evaluate_professional_safety,
    is_professional_occasion,
    normalize_professional_occasion,
)
from brain.engines.outfit_quality_guard import is_complete_board

logger = logging.getLogger("ahvi.constrained_outfit_builder")

GARMENT_SLOTS = ("top", "bottom", "dress", "outerwear", "footwear")
ALL_SLOTS = GARMENT_SLOTS + ("accessory",)

ERROR_CODES = frozenset({
    "INVALID_ITEM_ID",
    "INVALID_ANCHOR_ITEM",
    "UNKNOWN_ITEM_SOURCE",
    "INVALID_SLOT",
    "STYLE_ASSET_POOL_EMPTY",
    "ALL_ITEMS_LOCKED",
    "NO_REPLACEMENT_FOUND",
    "INSUFFICIENT_WARDROBE",
    "NO_VALID_COMBINATION",
    "FIXED_ITEMS_INCOMPATIBLE",
    "FIXED_ITEM_LOST",
    "EXCLUDED_ITEM_RETURNED",
    "CONSTRAINED_BUILDER_FAILED",
    "BOARD_SOURCE_POLICY_UNKNOWN",
    "SOURCE_POLICY_VIOLATION",
})

_SCENARIO_DEFAULT_SOURCES = {
    "build_outfit": ("wardrobe",),
    "style_this": ("wardrobe",),
}

# Footwear that should never lead a dress look (mirrors routers.stylist).
_BAD_DRESS_FOOTWEAR_TOKENS = (
    "loafer", "oxford", "derby", "brogue", "monk", "wingtip",
    "dress shoe", "formal shoe", "leather shoe", "leather shoes",
    "men's", "mens", "combat boot",
)

_NEUTRAL_COLORS = frozenset({
    "white", "black", "grey", "gray", "beige", "cream", "navy", "tan",
    "khaki", "ivory", "charcoal", "denim", "brown", "off-white", "offwhite",
})


def _occ_key(occasion: Optional[str]) -> str:
    return normalize_professional_occasion(occasion)

def _stable_offset(seed: str, size: int) -> int:
    """Deterministic rotation offset (same pattern as brain.outfit_pipeline)."""
    if size <= 0:
        return 0
    digest = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % size


def _failure(code: str, message: str, fixed_item_ids: Optional[List[str]] = None,
             **extra: Any) -> Dict[str, Any]:
    assert code in ERROR_CODES, f"unregistered failure code: {code}"
    error: Dict[str, Any] = {"code": code, "message": message}
    if fixed_item_ids is not None:
        error["fixed_item_ids"] = list(fixed_item_ids)
    error.update(extra)
    return {"success": False, "error": error}


class ConstrainedOutfitError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.details = dict(details or {})
        super().__init__(message)


def replaceable_slots_for_fixed_items(fixed_items: Sequence[Dict[str, Any]]) -> List[str]:
    """Return explicit legal completion slots for direct item actions."""
    roles = {canonical_item_role(item) for item in fixed_items}
    if "unknown" in roles:
        raise ConstrainedOutfitError("INVALID_SLOT", "A fixed item has no valid garment slot.")
    if "dress" in roles and ({"top", "bottom"} & roles):
        raise ConstrainedOutfitError(
            "FIXED_ITEMS_INCOMPATIBLE",
            "A fixed dress cannot be combined with a fixed top or bottom.",
        )
    if "dress" in roles:
        return ["footwear", "outerwear", "accessory"]
    if "top" in roles or "bottom" in roles:
        slots = ["top", "bottom", "footwear", "outerwear", "accessory"]
        return [slot for slot in slots if slot not in roles]
    if "footwear" in roles or "outerwear" in roles or "accessory" in roles:
        # Use the top+bottom structure deterministically. A future request-level
        # structure choice may select dress instead, but never both at once.
        return ["top", "bottom", "footwear", "outerwear", "accessory"]
    return ["top", "bottom", "dress", "footwear", "outerwear", "accessory"]


def _blob(item: Dict[str, Any]) -> str:
    return " ".join(
        str(item.get(k) or "")
        for k in ("name", "label", "category", "sub_category", "subcategory", "type")
    ).lower()


def _is_bad_dress_footwear(raw: Dict[str, Any]) -> bool:
    return any(t in _blob(raw) for t in _BAD_DRESS_FOOTWEAR_TOKENS)


def _strategy_avoid_match(item: Dict[str, Any], strategy: Dict[str, Any]) -> bool:
    meta = item.get("style_metadata") if isinstance(item.get("style_metadata"), dict) else {}
    blob = " ".join(
        [
            _blob(item),
            str(item.get("style") or ""),
            str(item.get("pattern") or ""),
            str(item.get("tags") or ""),
            str(meta.get("style_keywords") or ""),
        ]
    ).lower()
    return any(
        str(token or "").strip().lower() in blob
        for token in strategy.get("avoid") or []
        if str(token or "").strip()
    )


def _formality_value(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "homewear": 1.0, "athletic": 2.0, "casual": 3.0,
        "smart_casual": 5.0, "business_casual": 6.0,
        "formal": 8.0, "business_formal": 8.0,
    }.get(key)


class ConstrainedOutfitBuilder:
    """Builds outfits around immutable fixed items with typed failures."""

    def generate(
        self,
        scenario: str,
        fixed_items: Optional[Sequence[Dict[str, Any]]] = None,
        replaceable_slots: Optional[Sequence[str]] = None,
        exclude_item_ids: Optional[Sequence[str]] = None,
        source_policy: Any = None,
        context: Optional[Dict[str, Any]] = None,
        wardrobe: Optional[Sequence[Dict[str, Any]]] = None,
        style_assets: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        context = dict(context or {})
        scenario = str(scenario or "build_outfit").strip().lower()
        occasion = context.get("occasion")
        professional = is_professional_occasion(occasion)
        wardrobe_raw = [i for i in (wardrobe if wardrobe is not None else context.get("wardrobe") or []) if isinstance(i, dict)]
        style_asset_raw = [i for i in (style_assets if style_assets is not None else context.get("style_assets") or []) if isinstance(i, dict)]
        candidate_raw = wardrobe_raw + style_asset_raw
        variant = int(context.get("variant") or 0)
        prefer_footwear = tuple(context.get("prefer_footwear") or ())
        accessory_budget = min(2, max(0, int(context.get("accessory_budget", 1))))
        weather_context = context.get("weather_context") or context.get("weather")
        style_strategy = context.get("style_strategy") if isinstance(context.get("style_strategy"), dict) else {}

        # --- Fixed items: normalize + validate identity -------------------
        fixed_raw = [i for i in (fixed_items or []) if isinstance(i, dict)]
        fixed_norm: List[Dict[str, Any]] = []
        for raw in fixed_raw:
            norm = normalize_style_item(raw)
            if not norm["item_id"]:
                return _failure(
                    "INVALID_ITEM_ID",
                    "A locked item is missing a usable item id, so it cannot be preserved safely.",
                    fixed_item_ids=[canonical_item_id(i) for i in fixed_raw],
                )
            norm["locked"] = True
            if norm["source"] == "unknown":
                return _failure(
                    "UNKNOWN_ITEM_SOURCE",
                    "A fixed item must include a trusted source.",
                    fixed_item_ids=[norm["item_id"]],
                )
            if norm["role"] == "unknown":
                return _failure(
                    "INVALID_ANCHOR_ITEM",
                    "The selected item does not have a valid garment slot.",
                    fixed_item_ids=[norm["item_id"]],
                )
            if professional:
                decision = evaluate_professional_safety(raw, occasion)
                if not decision["allowed"]:
                    return _failure(
                        "FIXED_ITEMS_INCOMPATIBLE",
                        "A fixed item is incompatible with the requested professional occasion.",
                        fixed_item_ids=[norm["item_id"]],
                        reason_code=decision["reason_code"],
                    )
            fixed_norm.append(norm)
        fixed_ids = {n["item_id"] for n in fixed_norm}
        fixed_pairs = list(zip(fixed_raw, fixed_norm))

        # --- Exclusions (fixed items always win over exclusion) -----------
        exclude_ids = {
            str(x).strip() for x in (exclude_item_ids or []) if str(x or "").strip()
        } - fixed_ids

        # --- Source policy -------------------------------------------------
        allowed_sources = self._resolve_sources(scenario, source_policy, fixed_norm)
        if allowed_sources is None:
            return _failure(
                "UNKNOWN_ITEM_SOURCE",
                "The requested source policy contains no recognizable sources.",
            )

        # --- Replaceable slot validation ------------------------------------
        requested_slots: Optional[List[str]] = None
        if replaceable_slots is not None:
            requested_slots = [str(s or "").strip().lower() for s in replaceable_slots]
            bad = [s for s in requested_slots if s not in ALL_SLOTS]
            if bad:
                return _failure(
                    "INVALID_SLOT",
                    f"Unknown slot(s): {', '.join(bad)}. Valid slots: {', '.join(ALL_SLOTS)}.",
                )

        # --- Fixed slot model ------------------------------------------------
        fixed_by_slot: Dict[str, Dict[str, Any]] = {}
        fixed_accessories: List[Dict[str, Any]] = []
        for _raw, norm in fixed_pairs:
            role = norm["role"]
            if role == "accessory":
                fixed_accessories.append(norm)
            elif role in GARMENT_SLOTS and role not in fixed_by_slot:
                fixed_by_slot[role] = norm
            # unknown-role or duplicate-role fixed items are still preserved in
            # output (fixed precedence), they just do not own a slot.

        has_fixed_dress = "dress" in fixed_by_slot
        has_fixed_top_or_bottom = "top" in fixed_by_slot or "bottom" in fixed_by_slot
        if has_fixed_dress and has_fixed_top_or_bottom:
            return _failure(
                "FIXED_ITEMS_INCOMPATIBLE",
                "A fixed dress cannot be combined with a fixed top or bottom.",
                fixed_item_ids=sorted(fixed_ids),
            )

        # --- Candidate pools (constrained BEFORE combination) ----------------
        pools: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {
            s: [] for s in ALL_SLOTS
        }
        seen_pool_ids: set = set()
        for raw in candidate_raw:
            cid = canonical_item_id(raw)
            if not cid or cid in exclude_ids or cid in fixed_ids or cid in seen_pool_ids:
                continue
            role = canonical_item_role(raw)
            if role not in ALL_SLOTS:
                continue  # unknown / non-fashion / sport-swim
            source = canonical_item_source(raw)
            if source not in allowed_sources:
                continue  # unknown source is NOT wardrobe
            # Prepare BEFORE gating: the prepared object is the exact
            # representation that will be serialized if selected, so
            # renderability must be checked on it, not on `raw` - otherwise
            # normalize_style_item()'s image_url fallback can manufacture a
            # masked_url alias after the gate already passed (the live
            # device bug on ce4ade1). style_asset items keep their own
            # Flutter resolver branch (590cc2e) and normalize_style_item()
            # doesn't carry every camelCase candidate alias through, so they
            # are gated on the untouched raw dict instead - exactly today's
            # behavior, unchanged.
            prepared = prepare_board_item(raw)
            gate_item = raw if source == "style_asset" else prepared
            if not is_board_renderable(gate_item):
                continue  # no genuine board-safe image - never select, never display later
            if style_strategy and _strategy_avoid_match(raw, style_strategy):
                continue
            if professional:
                decision = evaluate_professional_safety(raw, occasion)
                if not decision["allowed"]:
                    continue
            seen_pool_ids.add(cid)
            prepared["locked"] = False
            pools[role].append((raw, prepared))

        if "style_asset" in allowed_sources:
            pool_has_assets = any(
                n["source"] == "style_asset"
                for slot in ALL_SLOTS
                for (_r, n) in pools[slot]
            )
            pool_has_any = any(pools[slot] for slot in ALL_SLOTS)
            asset_only = "wardrobe" not in allowed_sources
            if not pool_has_assets and (
                asset_only or (scenario == "style_this" and not pool_has_any)
            ):
                return _failure(
                    "STYLE_ASSET_POOL_EMPTY",
                    "No eligible style assets are available to complete this look.",
                    fixed_item_ids=sorted(fixed_ids),
                )

        # --- Slots to generate -------------------------------------------------
        slots_needed = self._slots_needed(
            fixed_by_slot, fixed_accessories, pools, requested_slots,
            accessory_budget, context,
        )

        if requested_slots is not None and not slots_needed and scenario == "shuffle_unlocked":
            return _failure(
                "ALL_ITEMS_LOCKED",
                "Every item on this board is locked - unlock at least one piece to shuffle.",
                fixed_item_ids=sorted(fixed_ids),
            )

        if not fixed_norm and not any(pools[s] for s in ALL_SLOTS):
            return _failure(
                "INSUFFICIENT_WARDROBE",
                "There are not enough eligible pieces to build a look yet.",
            )

        # --- Fill slots (fixed slots pool == [fixed item] exactly) --------------
        outfit_items: List[Dict[str, Any]] = list(fixed_norm)  # fixed first
        changed_slots: List[str] = []
        missing_slots: List[str] = []
        dress_in_outfit = has_fixed_dress
        seed_base = "|".join(sorted(fixed_ids)) or scenario

        for slot in slots_needed:
            picked = self._pick(
                pools, slot,
                dress_in_outfit=dress_in_outfit or ("dress" in slots_needed and slot == "footwear"),
                anchor_items=fixed_norm + [n for n in outfit_items if not n.get("locked")],
                prefer=prefer_footwear if slot == "footwear" else (),
                variant=variant,
                seed=f"{seed_base}:{slot}",
                occasion=occasion,
                weather_context=weather_context,
                style_strategy=style_strategy,
            )
            if picked is None:
                missing_slots.append(slot)
                continue
            raw, prepared = picked
            # `prepared` was already built (and gated) by prepare_board_item()
            # when the pool was constructed - the same object proven safe is
            # the one served, with no further transformation that could
            # manufacture a new image_url alias after the fact.
            norm = dict(prepared)
            norm["locked"] = False
            outfit_items.append(norm)
            changed_slots.append(slot)
            if norm["role"] == "dress":
                dress_in_outfit = True
            # remove from pool so accessories/slots never duplicate
            pools[norm["role"]] = [
                (r, n) for (r, n) in pools[norm["role"]]
                if n["item_id"] != norm["item_id"]
            ]
            # Invariant: candidate outfit still preserves every fixed item.
            try:
                assert_fixed_items_preserved(fixed_norm, outfit_items, stage=f"fill:{slot}")
            except FixedItemLostError as exc:
                logger.error("constrained_builder invariant failed: %s", exc)
                return _failure(
                    "FIXED_ITEM_LOST",
                    "A locked item was lost while building the look; nothing was changed.",
                    fixed_item_ids=exc.missing_ids,
                    stage=exc.stage,
                )

        if scenario == "shuffle_unlocked" and requested_slots and not changed_slots:
            return _failure(
                "NO_REPLACEMENT_FOUND",
                "No suitable replacements were found for the unlocked pieces.",
                fixed_item_ids=sorted(fixed_ids),
            )

        if not fixed_norm and not changed_slots:
            return _failure(
                "INSUFFICIENT_WARDROBE",
                "There are not enough eligible pieces to build a look yet.",
            )

        # --- Final invariant on the selected outfit ------------------------------
        try:
            assert_fixed_items_preserved(fixed_norm, outfit_items, stage="final")
        except FixedItemLostError as exc:
            logger.error("constrained_builder final invariant failed: %s", exc)
            return _failure(
                "FIXED_ITEM_LOST",
                "A locked item was lost during final assembly; nothing was changed.",
                fixed_item_ids=exc.missing_ids,
                stage=exc.stage,
            )

        if scenario == "style_this" and not is_complete_board(outfit_items):
            return _failure(
                "INSUFFICIENT_WARDROBE",
                "There are not enough wardrobe pieces to build a complete Style This look.",
                fixed_item_ids=sorted(fixed_ids),
            )

        return {
            "success": True,
            "scenario": scenario,
            "occasion": occasion,
            "source": {"allowed_sources": sorted(allowed_sources)},
            "items": outfit_items,
            "changed_slots": changed_slots,
            "missing_items": missing_slots,
        }

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _resolve_sources(
        self,
        scenario: str,
        source_policy: Any,
        fixed_norm: List[Dict[str, Any]],
    ) -> Optional[frozenset]:
        if isinstance(source_policy, dict):
            allowed = {
                str(s or "").strip().lower()
                for s in (
                    source_policy.get("allowed_completion_sources")
                    or source_policy.get("allowed_sources")
                    or []
                )
            }
            allowed = {s for s in allowed if s in VALID_SOURCES and s != "unknown"}
            return frozenset(allowed) if allowed else None
        if scenario in _SCENARIO_DEFAULT_SOURCES:
            return frozenset(_SCENARIO_DEFAULT_SOURCES[scenario])
        # No explicit policy: never infer from the fixed items' sources (a
        # wardrobe anchor in a Style This board must not force wardrobe-only).
        # The shuffle service always resolves and passes an explicit policy;
        # this is the standalone/legacy default.
        return frozenset({"wardrobe"})

    def _slots_needed(
        self,
        fixed_by_slot: Dict[str, Dict[str, Any]],
        fixed_accessories: List[Dict[str, Any]],
        pools: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]],
        requested_slots: Optional[List[str]],
        accessory_budget: int,
        context: Dict[str, Any],
    ) -> List[str]:
        has_fixed_dress = "dress" in fixed_by_slot
        has_fixed_top_or_bottom = "top" in fixed_by_slot or "bottom" in fixed_by_slot

        if requested_slots is not None:
            # Shuffle mode: generate exactly the requested slots, minus any
            # that are fixed, respecting the dress rules.
            out: List[str] = []
            for slot in requested_slots:
                if slot in fixed_by_slot:
                    continue  # fixed slot pool == [fixed item]; nothing to replace
                if slot == "dress" and has_fixed_top_or_bottom:
                    continue  # never add a dress next to a fixed top/bottom
                if slot in ("top", "bottom") and has_fixed_dress:
                    continue  # never add top/bottom over a fixed dress
                if slot == "accessory" and out.count("accessory") >= max(accessory_budget, 1):
                    continue
                out.append(slot)
            return out

        # Build/style modes: base slots = complement of fixed slots.
        if has_fixed_dress:
            base = ["dress", "footwear"]
        elif has_fixed_top_or_bottom:
            base = ["top", "bottom", "footwear"]
        else:
            # Anchor is footwear / outerwear / accessory / none: choose a hero
            # garment route from what the wardrobe can actually supply.
            if pools["dress"] and not (pools["top"] and pools["bottom"]):
                base = ["dress", "footwear"]
            elif pools["top"] or pools["bottom"] or not pools["dress"]:
                base = ["top", "bottom", "footwear"]
            else:
                base = ["dress", "footwear"]

        slots = [s for s in base if s not in fixed_by_slot]
        if context.get("include_outerwear") and "outerwear" not in fixed_by_slot:
            slots.append("outerwear")
        # Accessories budget (<=2) excluding fixed accessories.
        for _ in range(accessory_budget):
            slots.append("accessory")
        return slots

    def _pick(
        self,
        pools: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]],
        slot: str,
        *,
        dress_in_outfit: bool,
        anchor_items: List[Dict[str, Any]],
        prefer: Tuple[str, ...],
        variant: int,
        seed: str,
        occasion: Optional[str],
        weather_context: Optional[Dict[str, Any]] = None,
        style_strategy: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        cands = list(pools.get(slot) or [])
        if slot == "footwear" and dress_in_outfit:
            cands = [(r, n) for (r, n) in cands if not _is_bad_dress_footwear(r)]
        if not cands:
            return None

        anchor_colors = {
            str(n.get("color") or "").strip().lower()
            for n in anchor_items
            if str(n.get("color") or "").strip()
        }
        occ = _occ_key(occasion)

        def _score(pair: Tuple[Dict[str, Any], Dict[str, Any]]) -> float:
            raw, norm = pair
            score = 0.0
            if prefer:
                blob = _blob(raw)
                if any(p in blob for p in prefer):
                    score += 10.0
            color = str(norm.get("color") or "").strip().lower()
            if color:
                if color in _NEUTRAL_COLORS:
                    score += 2.0  # neutrals pair with everything
                if color in anchor_colors:
                    score += 1.5  # simple harmony echo
            if occ:
                tags = raw.get("occasion_tags") or raw.get("occasions") or raw.get("tags")
                if isinstance(tags, (list, tuple, set)):
                    tag_blob = " ".join(str(t) for t in tags).lower().replace(" ", "_")
                    if occ in tag_blob:
                        score += 3.0
            if isinstance(weather_context, dict):
                # Keep fixed-anchor completion aligned with the canonical
                # weather scorer. Import lazily to avoid the reasoning-engine
                # <-> builder module cycle.
                try:
                    from services.style_reasoning_engine import asset_weather_compatibility_score
                    score += float(asset_weather_compatibility_score(raw, {"weather_context": weather_context}))
                except Exception:
                    # Missing optional weather metadata is neutral.
                    pass
            strategy = style_strategy if isinstance(style_strategy, dict) else {}
            palette = {
                str(value or "").strip().lower()
                for value in strategy.get("palette") or []
                if str(value or "").strip()
            }
            if palette and color:
                if any(color == p or color in p or p in color for p in palette):
                    score += 3.0
                elif color in _NEUTRAL_COLORS:
                    score += 0.5
            strategy_formality = _formality_value(strategy.get("formality"))
            raw_meta = raw.get("style_metadata") if isinstance(raw.get("style_metadata"), dict) else {}
            item_formality = _formality_value(raw.get("formality") or raw_meta.get("formality"))
            if strategy_formality is not None and item_formality is not None:
                score += max(0.0, 2.0 - abs(strategy_formality - item_formality) * 0.4)
            return score

        # Deterministic: sort by (score desc, id) then rotate by variant +
        # stable hash of the combo signature so shuffles produce variety
        # without randomness.
        cands.sort(key=lambda p: (-_score(p), p[1]["item_id"]))
        if prefer:
            # Preference-driven slots (footwear directions) always take the
            # best match so "Date Night" really prefers heels.
            return cands[0]
        if style_strategy:
            best_score = _score(cands[0])
            best = [pair for pair in cands if _score(pair) == best_score]
            offset = (variant + _stable_offset(seed, len(best))) % len(best)
            return best[offset]
        offset = (variant + _stable_offset(seed, len(cands))) % len(cands)
        return cands[offset]


constrained_outfit_builder = ConstrainedOutfitBuilder()
