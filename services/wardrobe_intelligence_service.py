import json
import re
from typing import Any, Dict, List, Optional


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v or "").strip()]
    if isinstance(value, (tuple, set)):
        return [str(v) for v in value if str(v or "").strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.replace("|", ",").split(",") if part.strip()]
    return []


def normalize_occasion(query: Any) -> str:
    q = str(query or "").lower().replace("-", "_")

    if any(x in q for x in ["beach", "resort", "pool"]):
        return "beach"

    if any(x in q for x in ["gym", "workout", "fitness", "training", "yoga", "running"]):
        return "gym"

    if any(x in q for x in ["rave", "club"]):
        return "rave"

    if any(x in q for x in ["cocktail"]):
        return "cocktail"

    if any(x in q for x in ["party", "house_party"]):
        return "house_party"

    if any(x in q for x in ["office", "work", "meeting"]):
        return "office"

    if any(x in q for x in ["date", "dinner", "date_night", "tonight"]):
        return "date"

    if any(x in q for x in ["temple", "pooja", "puja", "mandir", "temple_modest"]):
        return "temple"

    return "daily"


def enrich_wardrobe_item(item: dict) -> dict:
    item = item if isinstance(item, dict) else {}
    tags = (
        _as_list(item.get("tags"))
        + _as_list(item.get("style_tags"))
        + _as_list(item.get("occasions"))
        + _as_list(item.get("occasion_tags"))
    )
    colors = _as_list(item.get("colors")) or _as_list(item.get("color")) or _as_list(item.get("color_code"))
    text = " ".join([
        str(item.get("name", "")),
        str(item.get("category", "")),
        str(item.get("subcategory") or item.get("sub_category") or ""),
        " ".join(tags),
        " ".join(colors),
    ]).lower()

    meta = {
        "category": item.get("category"),
        "subcategory": item.get("subcategory") or item.get("sub_category"),
        "formality": "casual",
        "occasion_affinity": [],
        "season_affinity": ["all_season"],
        "avoid_for": [],
        "needs_review": False,
        "confidence": item.get("confidence", 0.7),
    }

    if any(x in text for x in ["belt"]):
        meta.update({
            "category": "Accessories",
            "subcategory": "Belt",
            "formality": "smart_casual",
            "occasion_affinity": ["daily", "office", "date", "coffee_run"],
            "avoid_for": ["gym", "beach", "rave"],
        })

    elif any(x in text for x in ["loafer", "oxford", "derby"]):
        meta.update({
            "category": "Footwear",
            "subcategory": "Loafers",
            "formality": "smart_casual",
            "occasion_affinity": ["office", "date", "cocktail"],
            "avoid_for": ["gym", "beach", "rave"],
        })

    elif any(x in text for x in ["suit", "blazer", "tie"]):
        meta.update({
            "category": item.get("category") or "Outerwear",
            "subcategory": "Formalwear",
            "formality": "formal",
            "occasion_affinity": ["office", "business_formal", "cocktail"],
            "avoid_for": ["gym", "beach", "rave", "house_party"],
        })

    elif any(x in text for x in ["linen"]):
        meta.update({
            "formality": "casual",
            "occasion_affinity": ["beach", "resort", "daily"],
            "season_affinity": ["summer", "spring"],
            "avoid_for": ["business_formal", "rave"],
        })

    elif any(x in text for x in ["jeans", "denim"]):
        meta.update({
            "category": "Bottoms",
            "subcategory": "Jeans",
            "formality": "casual",
            "occasion_affinity": ["daily", "coffee_run", "house_party"],
            "avoid_for": ["business_formal", "temple_formal"],
        })

    elif any(x in text for x in ["sandal", "slides", "flip flop", "slipper"]):
        meta.update({
            "category": "Footwear",
            "subcategory": "Sandals",
            "formality": "casual",
            "occasion_affinity": ["beach", "resort", "daily"],
            "season_affinity": ["summer", "spring"],
            "avoid_for": ["office", "business_formal", "cocktail"],
        })

    elif any(x in text for x in ["watch"]):
        meta.update({
            "category": "Accessories",
            "subcategory": "Watch",
            "formality": "smart_casual",
            "occasion_affinity": ["office", "date", "cocktail", "daily"],
            "avoid_for": ["gym", "beach"],
        })

    elif any(x in text for x in ["saree", "sari"]):
        meta.update({
            "category": "Dresses",
            "subcategory": "Saree",
            "formality": "ethnic_formal",
            "occasion_affinity": ["wedding", "festival", "temple", "traditional"],
            "avoid_for": ["gym", "beach", "rave"],
        })

    if not meta["occasion_affinity"]:
        meta["occasion_affinity"] = ["daily"]

    return meta


def _style_meta(item: dict) -> dict:
    if not isinstance(item, dict):
        return {}
    raw = item.get("style_metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def score_item_for_occasion(item: dict, occasion: Any) -> int:
    meta = _style_meta(item)
    score = 0
    normalized = normalize_occasion(occasion)

    affinity = meta.get("occasion_affinity") or []
    avoid = meta.get("avoid_for") or []

    if normalized in affinity:
        score += 3

    if normalized in avoid:
        score -= 10

    return score


def board_has_occasion_conflict(board: dict, occasion: Any) -> bool:
    if not isinstance(board, dict):
        return False
    normalized = normalize_occasion(occasion)
    items = list(board.get("items") or [])
    for key in ("top", "bottom", "dress", "shoes", "footwear", "outerwear"):
        value = board.get(key)
        if isinstance(value, dict):
            items.append(value)
    for item in items:
        meta = _style_meta(item)
        if normalized in (meta.get("avoid_for") or []):
            return True
    return False


# ===========================================================================
# Climate Metadata V1
#
# climate_profile describes the GARMENT (what it physically is), never
# whether it's suitable for today's weather. No date/month/weather/location
# input is used anywhere below.
#
# Each property is a compact [value, confidence, source] tuple. `source` is
# one of the authority codes below, highest first. A lower-authority source
# may never touch the value/source/confidence of a higher one — the only way
# a "u" (user_confirmed) tuple changes is another explicit user edit.
# ===========================================================================

CLIMATE_SOURCE_USER = "u"
CLIMATE_SOURCE_VISION = "v"
CLIMATE_SOURCE_DETERMINISTIC = "d"
CLIMATE_SOURCE_MODEL = "m"
CLIMATE_SOURCE_UNKNOWN = "x"

CLIMATE_CONFIDENCE_NONE = 0
CLIMATE_CONFIDENCE_LOW = 1
CLIMATE_CONFIDENCE_MEDIUM = 2
CLIMATE_CONFIDENCE_HIGH = 3

CLIMATE_UNKNOWN_VALUE = "unknown"
CLIMATE_PROFILE_VERSION = "v1"

_CLIMATE_AUTHORITY = {
    CLIMATE_SOURCE_USER: 4,
    CLIMATE_SOURCE_VISION: 3,
    CLIMATE_SOURCE_DETERMINISTIC: 2,
    CLIMATE_SOURCE_MODEL: 1,
    CLIMATE_SOURCE_UNKNOWN: 0,
}

# material is user_confirmed-only in V1: no producer may claim an exact
# fiber/material identity from visual or categorical evidence.
CLIMATE_NON_AUTOMATED_KEYS = {"material"}

PHYSICAL_OBSERVATION_KEYS = {
    "fabric_weight",
    "fabric_structure",
    "fit",
    "drape",
    "coverage_level",
    "lining",
    "surface_texture",
}

APPAREL_CLIMATE_KEYS = (
    "material",
    "fabric_weight",
    "breathability",
    "insulation",
    "coverage_level",
    "fit",
    "layering_role",
    "water_resistance",
    "fabric_structure",
    "drape",
    "lining",
    "surface_texture",
)

# material is a cross-garment property (leather boots, suede loafers, canvas
# sneakers, ...) — it belongs in both key sets. It is still never
# auto-derived (see CLIMATE_NON_AUTOMATED_KEYS below); only an explicit user
# edit may ever populate it, for apparel or footwear alike.
FOOTWEAR_CLIMATE_KEYS = (
    "material",
    "footwear_type",
    "coverage",
    "construction_weight",
    "breathability",
    "water_resistance",
    "activity_affinity",
)

_FOOTWEAR_CATEGORY_TOKENS = {"footwear", "shoe", "shoes"}
_FOOTWEAR_SUBCATEGORY_TOKENS = (
    "sandal", "sneaker", "trainer", "boot", "loafer", "oxford", "derby",
    "slipper", "flip flop", "flip-flop", "heel", "pump", "slide", "shoe",
)


def climate_unknown_tuple() -> List[Any]:
    return [CLIMATE_UNKNOWN_VALUE, CLIMATE_CONFIDENCE_NONE, CLIMATE_SOURCE_UNKNOWN]


def climate_authority(source: Any) -> int:
    return _CLIMATE_AUTHORITY.get(str(source or "").strip().lower(), 0)


def _is_valid_climate_tuple(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and str(value[2] or "").strip().lower() in _CLIMATE_AUTHORITY
    )


def get_climate_property(profile: Optional[Dict[str, Any]], key: str) -> List[Any]:
    """Single canonical read path. Missing profile/key == unknown, always."""
    if isinstance(profile, dict):
        value = profile.get(key)
        if _is_valid_climate_tuple(value):
            return list(value)
    return climate_unknown_tuple()


def merge_climate_value(existing: Any, candidate: Any) -> List[Any]:
    """Authority-ordered merge of one property for GENERIC/AUTOMATED
    evidence (deterministic, vision, model, carried-forward prior state).
    Order-independent: whichever tuple has STRICTLY higher authority always
    wins regardless of which side is `existing` vs `candidate`.

    Equal authority -> existing survives by default. This is deliberate:
    letting the later-computed value win on a tie would make re-enrichment/
    backfill/agent runs order-dependent (a fresh vision "v" pass could
    silently displace a previously-accepted vision "v" value, an agent
    re-run could oscillate between two equally-weak model_inferred guesses,
    etc). The one case where an equal-authority replacement is legitimate —
    the user deliberately editing an already-user_confirmed value — is NOT
    handled here; it goes through apply_user_climate_edit() instead, which
    is reserved for that one call path and refuses anything that isn't
    itself a genuine user_confirmed tuple.
    """
    valid_existing = _is_valid_climate_tuple(existing)
    valid_candidate = _is_valid_climate_tuple(candidate)
    if not valid_candidate:
        return list(existing) if valid_existing else climate_unknown_tuple()
    if not valid_existing:
        return list(candidate)
    if climate_authority(candidate[2]) > climate_authority(existing[2]):
        return list(candidate)
    return list(existing)


def merge_climate_profile(
    base: Optional[Dict[str, Any]], incoming: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generic, automated-evidence merge. Monotonic in authority only —
    never replaces an equal-authority value. Never use this for an explicit
    user correction of an already-user_confirmed property; use
    apply_user_climate_edit() for that instead."""
    result: Dict[str, Any] = {k: list(v) for k, v in (base or {}).items()}
    for key, candidate in (incoming or {}).items():
        if key in CLIMATE_NON_AUTOMATED_KEYS:
            # material may only ever enter/replace via an explicit user edit.
            if not _is_valid_climate_tuple(candidate) or candidate[2] != CLIMATE_SOURCE_USER:
                continue
        result[key] = merge_climate_value(result.get(key), candidate)
    return result


def apply_user_climate_edit(
    profile: Optional[Dict[str, Any]], key: str, tuple_value: Any
) -> Dict[str, Any]:
    """The ONLY path allowed to replace an existing tuple with another of
    EQUAL authority — reserved for an explicit, intentional user correction
    (e.g. the user edits material a second time: linen/u -> cotton/u).
    Generic automated merge (merge_climate_value/merge_climate_profile)
    never does this — see its docstring for why.

    `tuple_value` must itself be a valid user_confirmed ("u") tuple;
    anything else is rejected outright and the profile is returned
    unchanged for that key. This means an automated producer can never
    reach the equal-authority override merely by shaping its output like a
    user tuple — the override isn't a property of the data, it's a property
    of which function call sites are allowed to invoke (only genuine
    user-input call sites like update_item_labels ever call this)."""
    result = {k: list(v) for k, v in (profile or {}).items()}
    if not _is_valid_climate_tuple(tuple_value) or tuple_value[2] != CLIMATE_SOURCE_USER:
        return result
    result[key] = list(tuple_value)
    return result


def normalize_material_value(value: Any) -> str:
    """Whitespace/case normalization only — never changes semantic meaning."""
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def user_confirmed_material_tuple(value: Any) -> Optional[List[Any]]:
    text = normalize_material_value(value)
    if not text:
        return None
    return [text, CLIMATE_CONFIDENCE_HIGH, CLIMATE_SOURCE_USER]


def is_footwear_item(category: Any, sub_category: Any) -> bool:
    cat = str(category or "").strip().lower()
    if cat in _FOOTWEAR_CATEGORY_TOKENS:
        return True
    sub = str(sub_category or "").strip().lower()
    return any(token in sub for token in _FOOTWEAR_SUBCATEGORY_TOKENS)


def _climate_vision_blob(vision_evidence: Dict[str, Any]) -> str:
    parts = [
        vision_evidence.get("name"),
        vision_evidence.get("sub_category") or vision_evidence.get("subcategory"),
        vision_evidence.get("pattern"),
    ]
    return " ".join(str(p or "") for p in parts).lower()


def _climate_broad_blob(item: Dict[str, Any]) -> str:
    parts = [
        item.get("name"),
        item.get("category"),
        item.get("sub_category") or item.get("subcategory"),
        item.get("pattern"),
    ]
    return " ".join(str(p or "") for p in parts).lower()


# Literal, purely descriptive terms — never a functional/suitability claim
# ("waterproof", "breathable", "good for summer" are NOT here). The same
# term list backs both tiers: with positively-established image provenance
# it becomes vision_observed (confidence MEDIUM); over plain stored text
# with unknown/no provenance it can only ever be deterministic_derived
# (confidence LOW) — see extract_vision_observed_climate_properties and
# derive_deterministic_climate_properties.
def _literal_apparel_terms(blob: str) -> Dict[str, str]:
    terms: Dict[str, str] = {}
    if "sleeveless" in blob:
        terms.setdefault("coverage_level", "sleeveless")
    if "long sleeve" in blob or "full sleeve" in blob:
        terms.setdefault("coverage_level", "full_sleeve")
    if "short sleeve" in blob or "half sleeve" in blob:
        terms.setdefault("coverage_level", "short_sleeve")
    if "full coverage" in blob:
        terms.setdefault("coverage_level", "full")
    if "quilted" in blob:
        terms.setdefault("insulation", "likely_insulated")
    if "loose fit" in blob or "relaxed fit" in blob:
        terms.setdefault("fit", "loose")
    if "lightweight" in blob:
        terms.setdefault("fabric_weight", "light")
    if "bulky" in blob or "heavy construction" in blob:
        terms.setdefault("fabric_weight", "heavy")
    return terms


def _literal_footwear_terms(blob: str) -> Dict[str, str]:
    terms: Dict[str, str] = {}
    if "open toe" in blob or "open-toe" in blob:
        terms.setdefault("coverage", "open_toe")
    if "closed toe" in blob or "closed-toe" in blob:
        terms.setdefault("coverage", "closed_toe")
    return terms


def extract_vision_observed_climate_properties(
    vision_evidence: Optional[Dict[str, Any]], *, footwear: bool
) -> Dict[str, Any]:
    """Literal, genuinely image-grounded descriptive terms only.

    `vision_evidence` must be the current, positively-provenanced vision
    detector output for THIS exact item — never the persisted/stored item
    record, which may have been user-edited, normalized, or backfilled long
    after capture and so cannot prove where its text came from. Provenance
    is verified via a `label_source`/`source` field that literally names a
    vision detector (e.g. "vision:gemini_multi", matching the marker the
    capture pipeline already stamps on real per-item vision output). Absent
    that marker — including when vision_evidence is just an empty/None
    placeholder — nothing is treated as vision_observed; deterministic_
    derived is the correct fallback tier instead.
    """
    if not isinstance(vision_evidence, dict) or not vision_evidence:
        return {}
    provenance = str(
        vision_evidence.get("label_source") or vision_evidence.get("source") or ""
    ).strip().lower()
    if not provenance.startswith("vision"):
        return {}

    blob = _climate_vision_blob(vision_evidence)
    terms = _literal_footwear_terms(blob) if footwear else _literal_apparel_terms(blob)
    return {
        key: [value, CLIMATE_CONFIDENCE_MEDIUM, CLIMATE_SOURCE_VISION]
        for key, value in terms.items()
    }


def map_physical_garment_observations(
    observations: Any,
    *,
    min_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert validated physical observations into Climate Metadata evidence.

    Physical vision evidence uses the existing vision authority:
        [value, medium-confidence, "v"]

    Exact material is intentionally never produced here.
    """
    if not isinstance(observations, dict):
        return {}

    if min_confidence is None:
        from services.physical_garment_analysis_service import (
            PHYSICAL_ANALYSIS_MIN_CONFIDENCE,
        )

        min_confidence = PHYSICAL_ANALYSIS_MIN_CONFIDENCE

    result: Dict[str, Any] = {}

    for key in PHYSICAL_OBSERVATION_KEYS:
        raw = observations.get(key)

        if not isinstance(raw, dict):
            continue

        value = raw.get("value")
        confidence = raw.get("confidence")

        if value == CLIMATE_UNKNOWN_VALUE:
            continue

        if value is None:
            continue

        try:
            confidence = float(confidence)
        except Exception:
            continue

        if not 0.0 <= confidence <= 1.0:
            continue

        if confidence < min_confidence:
            continue

        result[key] = [
            str(value),
            CLIMATE_CONFIDENCE_MEDIUM,
            CLIMATE_SOURCE_VISION,
        ]

    return result


def derive_deterministic_climate_properties(
    item: Dict[str, Any], *, footwear: bool
) -> Dict[str, Any]:
    """Conservative category/sub_category/name/pattern reasoning over the
    STORED item record — used both as the general low-confidence tier and
    as the fallback for literal descriptive terms whose provenance cannot be
    proven (e.g. a name that merely contains "sleeveless"/"quilted" with no
    accompanying vision evidence). Never fabricates an exact material, never
    uses date/month/weather/location."""
    item = item if isinstance(item, dict) else {}
    blob = _climate_broad_blob(item)
    out: Dict[str, Any] = {}

    literal = _literal_footwear_terms(blob) if footwear else _literal_apparel_terms(blob)
    for key, value in literal.items():
        out[key] = [value, CLIMATE_CONFIDENCE_LOW, CLIMATE_SOURCE_DETERMINISTIC]

    def set_if(key: str, present: bool, value: str) -> None:
        if present and key not in out:
            out[key] = [value, CLIMATE_CONFIDENCE_LOW, CLIMATE_SOURCE_DETERMINISTIC]

    if footwear:
        set_if("footwear_type", any(w in blob for w in ("sandal", "slide", "flip flop", "flip-flop")), "sandal")
        set_if("footwear_type", any(w in blob for w in ("sneaker", "trainer")), "sneaker")
        set_if("footwear_type", "boot" in blob, "boot")
        set_if("footwear_type", any(w in blob for w in ("loafer", "oxford", "derby")), "formal_shoe")
        set_if("footwear_type", any(w in blob for w in ("heel", "pump")), "heel")

        set_if("coverage", any(w in blob for w in ("sandal", "slide", "flip flop", "flip-flop")), "open")
        set_if("coverage", any(w in blob for w in ("boot", "sneaker", "trainer", "loafer", "oxford", "derby")), "closed")

        set_if("construction_weight", "boot" in blob, "heavy")
        set_if("construction_weight", any(w in blob for w in ("sandal", "slide", "flip flop", "flip-flop")), "light")

        # Open-toe/strap construction is direct construction evidence for
        # breathability — distinct from "fabric is lightweight" (Correction
        # 3 removed that inference; this one is about venting, not weight).
        set_if("breathability", any(w in blob for w in ("sandal", "slide", "flip flop", "flip-flop")), "likely_breathable")
        set_if("water_resistance", any(w in blob for w in ("rain boot", "waterproof", "rain shoe")), "likely_water_resistant")
        set_if("activity_affinity", any(w in blob for w in ("running", "trainer", "sneaker", "sports shoe")), "athletic")
        return out

    set_if("insulation", any(w in blob for w in ("puffer", "padded", "parka", "down jacket")), "likely_insulated")
    set_if("fabric_weight", any(w in blob for w in ("puffer", "padded", "parka", "overcoat", "wool coat")), "heavy")
    set_if("coverage_level", any(w in blob for w in ("tank top", "camisole")), "sleeveless")
    category = str(item.get("category") or "").strip().lower()
    set_if("layering_role", category == "outerwear", "outer_layer")
    set_if("layering_role", category == "innerwear", "base_layer")
    set_if("water_resistance", any(w in blob for w in ("raincoat", "windbreaker", "waterproof jacket")), "likely_water_resistant")
    # NOTE (Correction 3): apparel breathability is intentionally NOT derived
    # from fabric_weight alone — a lightweight garment can still be a
    # non-breathable coated/laminated synthetic. No apparel breathability
    # rule exists in V1; it stays unknown absent stronger evidence.
    return out


def normalize_agent_climate_profile(raw: Any) -> Dict[str, Any]:
    """Normalize an optional agent-reported climate_profile. Agent output is
    always demoted to model_inferred regardless of what it claims — an
    external model call is the lowest non-unknown authority, and it must
    never be able to assert a material identity."""
    if not isinstance(raw, dict):
        return {}
    allowed = (set(APPAREL_CLIMATE_KEYS) | set(FOOTWEAR_CLIMATE_KEYS)) - CLIMATE_NON_AUTOMATED_KEYS
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        key = str(key or "").strip()
        if key not in allowed:
            continue
        text_value = value[0] if isinstance(value, (list, tuple)) and value else value
        text_value = str(text_value or "").strip()
        if not text_value or text_value.lower() == CLIMATE_UNKNOWN_VALUE:
            continue
        out[key] = [text_value, CLIMATE_CONFIDENCE_LOW, CLIMATE_SOURCE_MODEL]
    return out


def fetch_existing_climate_profile(item_id: str) -> Dict[str, Any]:
    """Best-effort read of a previously-persisted climate_profile so it can
    be carried forward and merged (never raises)."""
    raw_id = str(item_id or "").strip()
    if not raw_id:
        return {}
    try:
        from services.appwrite_proxy import AppwriteProxy

        proxy = AppwriteProxy()
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_id).strip("._-")[:36]
        candidates = [raw_id] + ([safe_id] if safe_id and safe_id != raw_id else [])
        for doc_id in candidates:
            try:
                doc = proxy.get_document("wardrobe_style_metadata", doc_id)
            except Exception:
                continue
            raw = doc.get("style_metadata") if isinstance(doc, dict) else None
            parsed = raw
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = None
            climate_profile = parsed.get("climate_profile") if isinstance(parsed, dict) else None
            if isinstance(climate_profile, dict) and climate_profile:
                return climate_profile
    except Exception:
        return {}
    return {}


def build_climate_profile(
    item: Dict[str, Any],
    *,
    vision_evidence: Optional[Dict[str, Any]] = None,
    existing_profile: Optional[Dict[str, Any]] = None,
    physical_observations: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pure deterministic climate_profile producer for one garment.

    Never uses current date/month/weather/location. Combines (in
    authority-safe, order-independent fashion): prior evidence carried
    forward, fresh deterministic derivation over the stored item record,
    then fresh vision-observed extraction — but ONLY when `vision_evidence`
    is a positively-provenanced, current vision detector output (see
    extract_vision_observed_climate_properties). Passing the stored item
    itself as `vision_evidence` is intentionally NOT how this works — that
    text could have been user-edited or normalized long after capture, so
    it can only ever feed the deterministic tier. A caller may layer an
    explicit user material tuple or an optional agent contribution on top
    via merge_climate_profile.
    """
    item = item if isinstance(item, dict) else {}
    footwear = is_footwear_item(item.get("category"), item.get("sub_category") or item.get("subcategory"))
    keys = FOOTWEAR_CLIMATE_KEYS if footwear else APPAREL_CLIMATE_KEYS

    profile: Dict[str, Any] = {k: climate_unknown_tuple() for k in keys}
    if existing_profile:
        profile = merge_climate_profile(
            profile, {k: v for k, v in existing_profile.items() if k in keys}
        )
    profile = merge_climate_profile(profile, derive_deterministic_climate_properties(item, footwear=footwear))
    profile = merge_climate_profile(
        profile, extract_vision_observed_climate_properties(vision_evidence, footwear=footwear)
    )

    # 4. Add physical garment observations from dedicated physical analysis service.
    physical_properties = map_physical_garment_observations(physical_observations)
    for key, candidate in physical_properties.items():
        if key in keys:
            profile[key] = merge_climate_value(profile.get(key), candidate)

    return profile
