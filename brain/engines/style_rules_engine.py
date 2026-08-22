from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _text(value: Any) -> str:
    if isinstance(value, dict):
        fields = [
            "id",
            "name",
            "label",
            "title",
            "type",
            "category",
            "cat",
            "slot",
            "role",
            "subcategory",
            "sub_category",
            "garment_type",
            "description",
            "style",
            "fabric",
            "occasion",
            "color",
        ]
        return " ".join(str(value.get(k, "") or "") for k in fields).lower()
    return str(value or "").lower()


def _tokens(value: Any) -> set:
    return set(re.sub(r"[^a-z0-9]+", " ", _text(value)).split())


class StyleEngine:
    """
    AHVI deterministic style rules engine.

    This file is the single source for:
    - identity safety
    - occasion safety
    - footwear safety
    - scoring boosts / penalties
    - accessory duplication logic
    """

    FEMALE_ONLY = {
        "sports_bra",
        "bra",
        "bralette",
        "bustier",
        "saree",
        "sari",
        "lehenga",
        "gown",
        "dress",
        "women_dress",
        "skirt",
        "heels",
        "heel",
        "blouse",
        "kurti",
    }

    MALE_ONLY = {
        "sherwani",
    }

    DATE_BLOCK = {
        "slipper",
        "slippers",
        "slider",
        "sliders",
        "flipflop",
        "flip_flop",
        "crocs",
        "running_shoes",
        "gym_shoes",
        "sports_shoes",
    }

    DATE_BOOST = {
        "loafer",
        "loafers",
        "chelsea_boots",
        "boots",
        "clean_sneakers",
        "white_sneakers",
        "leather_sneakers",
        "watch",
        "belt",
    }

    WORKOUT_BLOCK = {
        "formal_shirt",
        "dress_shirt",
        "blazer",
        "suit",
        "loafer",
        "loafers",
        "formal_shoes",
        "heels",
        "belt",
        "saree",
        "lehenga",
        "dress",
    }

    WORKOUT_BOOST = {
        "dryfit",
        "dri_fit",
        "performance",
        "training",
        "gym",
        "running",
        "athletic",
        "track",
        "shorts",
        "joggers",
        "sneakers",
        "running_shoes",
        "training_shoes",
    }

    BUSINESS_BLOCK = {
        "slipper",
        "slippers",
        "slider",
        "sliders",
        "crocs",
        "sports_bra",
        "bra",
        "gym",
        "running_shoes",
        "beachwear",
    }

    BUSINESS_BOOST = {
        "shirt",
        "formal_shirt",
        "polo",
        "trouser",
        "trousers",
        "chinos",
        "blazer",
        "loafer",
        "loafers",
        "belt",
        "watch",
        "navy",
        "black",
        "white",
        "beige",
        "grey",
        "gray",
    }

    TRAVEL_BOOST = {
        "linen",
        "cotton",
        "breathable",
        "relaxed",
        "sneakers",
        "sandals",
        "overshirt",
        "jacket",
        "cap",
        "sunglasses",
        "bag",
    }

    ACCESSORY_TYPES = {
        "watch": {"watch", "watches"},
        "belt": {"belt", "belts"},
        "cap": {"cap", "caps", "hat", "hats"},
        "eyewear": {"sunglass", "sunglasses", "eyewear", "glasses"},
        "bag": {"bag", "bags", "backpack", "tote", "crossbody"},
        "jewelry": {"jewelry", "jewellery", "ring", "rings", "necklace", "bracelet", "earring"},
    }

    def normalize_gender(self, value: Any) -> str:
        v = _norm(value)
        if v in {"male", "man", "men", "boy", "m"}:
            return "male"
        if v in {"female", "woman", "women", "girl", "f"}:
            return "female"
        return "unknown"

    def normalize_intent(self, value: Any) -> str:
        t = _text(value)

        if any(x in t for x in ["date", "dinner", "night out"]):
            return "date_night"

        if any(x in t for x in ["workout", "gym", "fitness", "run", "running", "training"]):
            return "workout"

        if any(x in t for x in ["business", "office", "meeting", "work"]):
            return "business"

        if any(x in t for x in ["goa", "beach", "vacation", "trip", "travel", "airport"]):
            return "travel"

        if any(x in t for x in ["camp", "camping", "hiking", "trek", "trail"]):
            return "outdoor"

        return _norm(value) or "daily"

    def get_scoring_rules(self, style_dna: Dict[str, Any] | None, context: Dict[str, Any] | None):
        style_dna = style_dna or {}
        context = context or {}

        user_profile = (
            context.get("user_profile")
            or context.get("profile")
            or context.get("user")
            or {}
        )

        gender = self.normalize_gender(
            user_profile.get("gender")
            or context.get("gender")
            or style_dna.get("gender")
            or style_dna.get("gender_profile")
        )

        intent = self.normalize_intent(
            context.get("occasion")
            or context.get("intent")
            or context.get("query")
            or context.get("user_query")
        )

        preferred_colors = [
            str(c).lower()
            for c in style_dna.get("preferred_colors", []) or []
        ]

        avoided_items = {
            _norm(x)
            for x in style_dna.get("avoided_items", []) or []
            if x
        }

        boost_items = {
            _norm(x)
            for x in style_dna.get("preferred_styles", []) or []
            if x
        }

        if gender == "male":
            avoided_items |= self.FEMALE_ONLY
        elif gender == "female":
            avoided_items |= self.MALE_ONLY

        if intent == "date_night":
            avoided_items |= self.DATE_BLOCK
            boost_items |= self.DATE_BOOST

        elif intent == "workout":
            avoided_items |= self.WORKOUT_BLOCK
            boost_items |= self.WORKOUT_BOOST

        elif intent == "business":
            avoided_items |= self.BUSINESS_BLOCK
            boost_items |= self.BUSINESS_BOOST

        elif intent == "travel":
            boost_items |= self.TRAVEL_BOOST

        return {
            "gender": gender,
            "intent": intent,
            "preferred_colors": preferred_colors,
            "avoided_items": sorted(avoided_items),
            "boost_items": sorted(boost_items),
            "max_accessories": 2,
            "max_same_accessory_type": 1,
        }

    def item_keys(self, item: Dict[str, Any]) -> set:
        toks = _tokens(item)
        keys = set(toks)

        if {"sports", "bra"}.issubset(toks):
            keys.add("sports_bra")

        if {"dress", "shirt"}.issubset(toks):
            keys.add("dress_shirt")
            keys.discard("dress")

        if {"running", "shoe"}.issubset(toks) or {"running", "shoes"}.issubset(toks):
            keys.add("running_shoes")

        if {"formal", "shoe"}.issubset(toks) or {"formal", "shoes"}.issubset(toks):
            keys.add("formal_shoes")

        if {"white", "sneakers"}.issubset(toks) or {"white", "sneaker"}.issubset(toks):
            keys.add("white_sneakers")

        if {"leather", "sneakers"}.issubset(toks) or {"clean", "sneakers"}.issubset(toks):
            keys.add("clean_sneakers")

        keys.add(_norm(item.get("type")))
        keys.add(_norm(item.get("category")))
        keys.add(_norm(item.get("subcategory") or item.get("sub_category")))
        keys.add(_norm(item.get("slot") or item.get("role")))

        return {k for k in keys if k}

    def is_blocked_item(self, item: Dict[str, Any], rules: Dict[str, Any]) -> Tuple[bool, str]:
        item_keys = self.item_keys(item)
        blocked = set(rules.get("avoided_items") or [])
        hit = item_keys.intersection(blocked)

        if hit:
            return True, sorted(hit)[0]

        return False, ""

    def score_item_rule_adjustment(self, item: Dict[str, Any], rules: Dict[str, Any]) -> Tuple[float, List[str]]:
        blocked, reason = self.is_blocked_item(item, rules)
        if blocked:
            return -100.0, [f"blocked: {reason}"]

        keys = self.item_keys(item)
        boosts = set(rules.get("boost_items") or [])

        score = 0.0
        reasons = []

        if keys.intersection(boosts):
            score += 0.8
            # Internal scoring reason only — surfaced to the user via the
            # safety-gate copy after a real threshold check. The generic
            # 'occasion appropriate' label was misleading when the
            # underlying outfit only matched on a single boost token.
            reasons.append("aligns_with_occasion")

        color = str(item.get("color") or "").lower()
        if color and color in set(rules.get("preferred_colors") or []):
            score += 0.5
            reasons.append("preferred color")

        return score, reasons

    def accessory_type(self, item: Dict[str, Any]) -> str:
        keys = self.item_keys(item)

        for name, variants in self.ACCESSORY_TYPES.items():
            if keys.intersection(variants):
                return name

        return "other"


style_engine = StyleEngine()