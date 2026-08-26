"""
services/trend_context_service.py
AHVI Trend Intelligence V1 -- curated, source-backed board annotation.

Adapted from PR #48 (feat/trend-intelligence-dynamic, f0a57eb) with the
following corrections made during the clean rebuild review:

1. No private occasion taxonomy. PR #48's `_CANONICAL_OCCASION_MAP` /
   `_canonicalize_occasion()` (including a `key in raw` substring fallback
   that misclassified "candidate_interview" as a date-night occasion) has
   been removed entirely. This service now only accepts an
   ALREADY-canonicalized `canonical_occasion` string, resolved upstream by
   AHVI's existing `brain.engines.style_scorer.normalize_occasion()`. It
   never reinterprets raw user text.

   NOTE (discovered during this rebuild, out of scope to fix here):
   `normalize_occasion()` itself has the same class of bug --
   `normalize_occasion("candidate_interview")` currently returns
   "date_night" because its own date-detection branch does a plain
   substring check (`"date" in readable`) and "date" is a literal substring
   of "candidate". This is a pre-existing bug in AHVI's core occasion
   canonicalizer, independent of trend intelligence, affecting every
   occasion-dependent code path that calls normalize_occasion() -- not just
   this service. It should be fixed in brain/engines/style_scorer.py as its
   own dedicated change with a full regression pass, not patched locally
   here (that would recreate exactly the parallel-taxonomy problem this
   rebuild was told to remove). Practical impact on Trend Intelligence V1
   specifically: none observable today, because VERIFIED_ACTIVE_TRENDS=0
   (see style_trend_registry.py) -- no trend can currently surface under
   any occasion, correct or not.

2. Canonical item taxonomy. Trend category matching goes through
   services.style_item_contract.canonical_item_role() instead of raw
   item.get("category") string checks (see TREND_CATEGORY_TO_AHVI_ROLE).

3. Verification gate. is_trend_valid() hard-requires
   verification_status == "VERIFIED" in addition to review_state ==
   "approved" and a parseable, non-future verified_at. No record in the
   current registry satisfies this (see style_trend_registry.py), so
   get_active_trends() returns [] until real verified data exists --
   correct, fail-safe V1 behavior, not a bug.

4. India/global policy. target_region="india" is the default. India-scoped
   trends are tried first; global trends are only considered when
   allow_global_fallback=True AND no India-scoped trend produced a scored
   match for the same board (see match_board_trend()).

5. Determinism. No set()-based list deduplication anywhere -- matched_item_ids
   uses dict.fromkeys() to dedupe while preserving first-seen order.

6. Claim contract. annotate_board_with_trend() classifies its own output as
   exactly one of SOURCE_BACKED_CURRENT / SOURCE_BACKED_RECENT /
   CURATED_CONTEXT / NO_VALID_TREND (see TrendClaim). No trend explanation
   is ever attached without a claim tag proving what kind of backing it has.
"""
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from services.style_trend_registry import get_trend_registry
from services.style_item_contract import canonical_item_role

MIN_BOARD_TREND_THRESHOLD = 0.55
MIN_TREND_CONFIDENCE = 0.60
MAX_SOURCE_AGE_DAYS = 365
MAX_VERIFICATION_AGE_DAYS = 180  # re-verification staleness window

logger = logging.getLogger("ahvi.services.trend_context")

# ---------------------------------------------------------------------------
# Claim contract (Section 11) -- what kind of backing a trend match has.
# ---------------------------------------------------------------------------
SOURCE_BACKED_CURRENT = "SOURCE_BACKED_CURRENT"   # verified source, published <=90d ago
SOURCE_BACKED_RECENT = "SOURCE_BACKED_RECENT"     # verified source, published <=365d ago
CURATED_CONTEXT = "CURATED_CONTEXT"               # valid+scored but no fresh verified source
NO_VALID_TREND = "NO_VALID_TREND"                 # no match; board unaffected

SOURCE_CURRENT_WINDOW_DAYS = 90

# ---------------------------------------------------------------------------
# Canonical item taxonomy mapping (Section 7). Trend registry "categories"
# values -> AHVI's canonical_item_role() vocabulary
# ({top,bottom,dress,outerwear,footwear,accessory,unknown}). Hand-reviewed,
# static lookup -- not a second taxonomy, just a translation table into the
# one that already exists. Unknown/unmapped trend category terms degrade
# safely: score_board_against_trend() simply won't get a category bonus for
# that item, it never blocks the keyword-based match.
# ---------------------------------------------------------------------------
TREND_CATEGORY_TO_AHVI_ROLE: Dict[str, str] = {
    "top": "top",
    "bottom": "bottom",
    "outerwear": "outerwear",
    "shoes": "footwear",
    "footwear": "footwear",
    "accessories": "accessory",
    "jewelry": "accessory",
    "bags": "accessory",
    "dresses": "dress",
    # "ethnic_wear" intentionally NOT mapped to a single role -- ethnic
    # garments span top/bottom/dress depending on the specific piece
    # (kurta -> top, lehenga -> dress, etc). Per-item canonical_item_role()
    # already resolves this correctly; forcing one category-level role here
    # would be wrong for some items. Degrades safely (no category bonus).
}


def _normalize_gender(gender_val: Optional[str]) -> str:
    """male/man/men/m -> 'male'; female/woman/women/w/f -> 'female'; else 'unisex'."""
    if not gender_val:
        return "unisex"
    g = str(gender_val).lower().strip()
    if g in {"male", "man", "men", "m"}:
        return "male"
    if g in {"female", "woman", "women", "w", "f"}:
        return "female"
    return "unisex"


def annotate_board_with_trend(
    board_dict: dict,
    user_gender: Optional[str] = None,
    canonical_occasion: Optional[str] = None,
    target_region: str = "india",
    allow_global_fallback: bool = False,
    target_date: Any = None,
) -> dict:
    """Additive soft trend annotation. Never mutates items/roles/anchor/source
    provenance/board completeness -- only ever adds/removes the top-level
    trend_label / trend_explanation / trend_meta keys.

    Wired into style_flow_service.finalize_style_response_payload() AFTER
    candidate generation, garment role validation, negative compatibility,
    occasion/dress-code safety, and board completeness have already run --
    this function receives the FINAL validated board, never influences how
    it got built. Fail-open: any error clears trend keys and returns the
    board unchanged; it never raises into the caller.

    `canonical_occasion` must already be resolved by the caller via AHVI's
    existing occasion canonicalizer -- this function does not interpret
    raw user text.
    """
    if not isinstance(board_dict, dict):
        return board_dict
    try:
        items = board_dict.get("items") or board_dict.get("garments") or []
        trend_match = TrendContextService.match_board_trend(
            board_items=items,
            gender=user_gender,
            canonical_occasion=canonical_occasion,
            target_region=target_region,
            allow_global_fallback=allow_global_fallback,
            target_date=target_date,
        )
        if trend_match:
            board_dict["trend_label"] = trend_match["label"]
            board_dict["trend_explanation"] = trend_match.get("explanation", "")
            board_dict["trend_claim"] = trend_match["claim"]
            board_dict["trend_meta"] = {
                "trend_id": trend_match["trend_id"],
                "match_score": trend_match["match_score"],
                "region": trend_match.get("region", target_region),
                "valid_until": trend_match.get("valid_until"),
                "confidence": trend_match.get("confidence", 0.0),
                "publisher": trend_match.get("publisher", ""),
                "published_at": trend_match.get("published_at", ""),
                "claim": trend_match["claim"],
            }
        else:
            board_dict.pop("trend_label", None)
            board_dict.pop("trend_explanation", None)
            board_dict.pop("trend_meta", None)
            board_dict.pop("trend_claim", None)
    except Exception as exc:  # noqa: BLE001 - fail open, board must still return.
        logger.warning(
            "trend_annotation.failed occasion=%s err=%s", canonical_occasion, exc, exc_info=False,
        )
        board_dict.pop("trend_label", None)
        board_dict.pop("trend_explanation", None)
        board_dict.pop("trend_meta", None)
        board_dict.pop("trend_claim", None)

    return board_dict


class TrendContextService:
    @staticmethod
    def _parse_date(target_date: Any) -> Optional[date]:
        if not target_date:
            return None
        if isinstance(target_date, datetime):
            return target_date.date()
        if isinstance(target_date, date):
            return target_date
        if isinstance(target_date, str):
            clean = target_date.strip().split("T")[0]
            if not clean:
                return None
            try:
                return datetime.fromisoformat(clean).date()
            except Exception:
                return None
        return None

    @classmethod
    def is_trend_valid(
        cls,
        trend: Dict[str, Any],
        target_date: Any = None,
        target_region: str = "india",
        allow_global_fallback: bool = False,
    ) -> bool:
        """Strict verification, freshness, provenance, confidence, and
        region gate. False if unverified, expired, unsourced, unpublished,
        future-published, stale, low confidence, unapproved, or wrong region."""
        if not isinstance(trend, dict):
            return False

        # 1. Verification gate (BLOCKING -- Section 3). A trend is never
        # active on manually-entered metadata alone.
        if str(trend.get("verification_status") or "").strip().upper() != "VERIFIED":
            return False

        # 2. Review state gate.
        rev = trend.get("review_state")
        if not rev or str(rev).lower().strip() != "approved":
            return False

        # 3. verified_at gate -- must exist, parse, and not be future-dated.
        current_d = cls._parse_date(target_date) or datetime.utcnow().date()
        verified_d = cls._parse_date(trend.get("verified_at"))
        if not verified_d or verified_d > current_d:
            return False
        if (current_d - verified_d).days > MAX_VERIFICATION_AGE_DAYS:
            return False

        # 4. Freshness window gate.
        v_from = cls._parse_date(trend.get("valid_from"))
        v_until = cls._parse_date(trend.get("valid_until"))
        if not v_from or not v_until:
            return False
        if not (v_from <= current_d <= v_until):
            return False

        # 5. Source & provenance gate.
        source = trend.get("source")
        if not isinstance(source, dict):
            return False
        publisher = str(source.get("publisher") or "").strip()
        url = str(source.get("url") or "").strip()
        pub_at = cls._parse_date(source.get("published_at"))
        if not publisher or not url or not pub_at:
            return False

        # 6. Future publication date gate.
        if pub_at > current_d:
            return False

        # 7. Stale source gate.
        if (v_from - pub_at).days > MAX_SOURCE_AGE_DAYS or (current_d - pub_at).days > MAX_SOURCE_AGE_DAYS:
            return False

        # 8. Confidence gate.
        try:
            conf = float(trend.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if conf < MIN_TREND_CONFIDENCE:
            return False

        # 9. Region & scope gate (India-first policy, Section 8).
        req_region = str(target_region or "india").lower().strip()
        trend_regions = [str(r).lower().strip() for r in (trend.get("region") or [])]
        trend_scope = str(trend.get("scope") or "").lower().strip()
        is_region_match = req_region in trend_regions or trend_scope == req_region
        if not is_region_match:
            is_global_trend = "global" in trend_regions or trend_scope == "global"
            if not (allow_global_fallback and is_global_trend):
                return False

        return True

    @classmethod
    def get_active_trends(
        cls,
        target_date: Any = None,
        gender: Optional[str] = None,
        canonical_occasion: Optional[str] = None,
        target_region: str = "india",
        allow_global_fallback: bool = False,
    ) -> List[Dict[str, Any]]:
        """Filter registry by verification/freshness/provenance gates, gender,
        and an ALREADY-canonical occasion (no reinterpretation of raw text)."""
        all_trends = get_trend_registry()
        active = []
        norm_req_gender = _normalize_gender(gender)
        canon_occ = str(canonical_occasion or "").strip().lower().replace(" ", "_") or None

        for trend in all_trends:
            if not cls.is_trend_valid(
                trend, target_date=target_date, target_region=target_region,
                allow_global_fallback=allow_global_fallback,
            ):
                continue

            if norm_req_gender != "unisex":
                t_genders = [_normalize_gender(g) for g in trend.get("gender", [])]
                if norm_req_gender not in t_genders and "unisex" not in t_genders:
                    continue

            if canon_occ:
                t_occs = [str(o).lower().strip().replace(" ", "_") for o in trend.get("occasions", [])]
                if canon_occ not in t_occs:
                    continue

            active.append(trend)

        return active

    @classmethod
    def get_diagnostics(
        cls, target_date: Any = None, target_region: str = "india",
    ) -> Dict[str, Any]:
        all_trends = get_trend_registry()
        current_d = cls._parse_date(target_date) or datetime.utcnow().date()
        expired_count = 0
        unverified_count = 0
        active_count = 0
        latest_ingested = ""

        for t in all_trends:
            ing = str(t.get("ingested_at") or "").strip()
            if ing > latest_ingested:
                latest_ingested = ing
            v_until = cls._parse_date(t.get("valid_until"))
            if v_until and v_until < current_d:
                expired_count += 1
            if str(t.get("verification_status") or "").strip().upper() != "VERIFIED":
                unverified_count += 1
            if cls.is_trend_valid(t, target_date=target_date, target_region=target_region):
                active_count += 1

        return {
            "trend_registry_loaded": len(all_trends) > 0,
            "total_record_count": len(all_trends),
            "active_record_count": active_count,
            "latest_ingestion": latest_ingested or "none",
            "expired_count": expired_count,
            "unverified_count": unverified_count,
            "current_region": target_region,
            "source_health": "healthy" if active_count > 0 else "degraded",
            "mode": "CURATED_SOURCE_BACKED",
        }

    @classmethod
    def score_board_against_trend(
        cls,
        board_items: List[Dict[str, Any]],
        trend: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate items individually via canonical role + free-text keyword
        signal. An item MUST have a strong keyword/tag match to qualify --
        color-only coincidence can never produce a false positive."""
        if not board_items:
            return None

        trend_categories = [c.lower() for c in trend.get("categories", [])]
        trend_ahvi_roles = {
            TREND_CATEGORY_TO_AHVI_ROLE[c] for c in trend_categories if c in TREND_CATEGORY_TO_AHVI_ROLE
        }
        trend_colors = [c.lower() for c in trend.get("colors", [])]
        trend_keywords = [k.lower() for k in trend.get("keywords", [])]

        matched_item_count = 0
        matched_item_ids: List[str] = []
        matched_item_names: List[str] = []
        matched_reasons: List[str] = []
        total_item_points = 0.0

        for item in board_items:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or item.get("item_id") or item.get("garment_id") or item.get("$id") or "")
            item_name = str(item.get("name") or item.get("title") or item.get("label") or "").strip()
            item_name_lower = item_name.lower()
            item_color = str(item.get("color") or item.get("colour") or "").lower()
            raw_tags = item.get("tags") or item.get("style_tags") or []
            item_tags = [str(t).lower() for t in (raw_tags if isinstance(raw_tags, list) else [])]
            item_subcat = str(item.get("sub_category") or item.get("subcategory") or item.get("type") or "").lower()

            item_text = f"{item_name_lower} {item_subcat} {' '.join(item_tags)}"

            matched_keywords = [kw for kw in trend_keywords if kw in item_text]
            if not matched_keywords:
                continue  # strong-signal requirement; color-only never matches

            item_score = 2.0  # base points for a genuine keyword/tag signal

            if any(c in item_color or c in item_name_lower for c in trend_colors):
                item_score += 1.0

            ahvi_role = canonical_item_role(item)
            if ahvi_role in trend_ahvi_roles:
                item_score += 0.5

            total_item_points += item_score
            matched_reasons.extend(matched_keywords)
            matched_item_count += 1
            if item_name:
                matched_item_names.append(item_name)
            if item_id:
                matched_item_ids.append(item_id)

        if matched_item_count == 0:
            return None

        avg_item_score = total_item_points / matched_item_count
        coverage = matched_item_count / len(board_items)
        normalized_score = (avg_item_score / 3.5) * coverage
        if coverage >= 0.5:
            normalized_score = min(normalized_score + 0.1, 1.0)
        if normalized_score < MIN_BOARD_TREND_THRESHOLD:
            return None

        label = str(trend.get("label") or "")
        unique_names = list(dict.fromkeys(matched_item_names))  # stable dedup, order preserved
        if len(unique_names) == 1:
            matched_phrase = unique_names[0].lower()
        elif len(unique_names) >= 2:
            matched_phrase = f"{', '.join(x.lower() for x in unique_names[:-1])} & {unique_names[-1].lower()}"
        else:
            matched_phrase = "existing pieces"

        source = trend.get("source") or {}
        claim, explanation = cls._classify_claim(trend, label, matched_phrase)

        return {
            "trend_id": trend.get("trend_id"),
            "label": label,
            "explanation": explanation,
            "claim": claim,
            "match_score": round(normalized_score, 2),
            "valid_until": trend.get("valid_until"),
            "confidence": float(trend.get("confidence", 0.0)),
            "publisher": str(source.get("publisher") or ""),
            "published_at": str(source.get("published_at") or ""),
            "matched_item_ids": list(dict.fromkeys(matched_item_ids)),  # stable dedup
            "matched_reasons": list(dict.fromkeys(matched_reasons))[:3],  # stable dedup
        }

    @classmethod
    def _classify_claim(cls, trend: Dict[str, Any], label: str, matched_phrase: str) -> tuple:
        """Section 11 claim contract. Only VERIFIED trends reach here (gated
        upstream by is_trend_valid), so this only ever distinguishes
        CURRENT vs RECENT by source age -- it never fabricates a
        SOURCE_BACKED claim for an unverified record."""
        source = trend.get("source") or {}
        pub_at = cls._parse_date(source.get("published_at"))
        today = datetime.utcnow().date()
        age_days = (today - pub_at).days if pub_at else None

        if age_days is not None and age_days <= SOURCE_CURRENT_WINDOW_DAYS:
            claim = SOURCE_BACKED_CURRENT
            explanation = f"This {label.lower()} direction is trending now, and your {matched_phrase} already fit it."
        elif age_days is not None:
            claim = SOURCE_BACKED_RECENT
            explanation = f"This {label.lower()} direction has been current recently, and your {matched_phrase} already fit it."
        else:
            claim = CURATED_CONTEXT
            explanation = f"This {label.lower()} direction fits your {matched_phrase}."
        return claim, explanation

    @classmethod
    def match_board_trend(
        cls,
        board_items: List[Dict[str, Any]],
        gender: Optional[str] = None,
        canonical_occasion: Optional[str] = None,
        target_region: str = "india",
        allow_global_fallback: bool = False,
        target_date: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """India-first, global-fallback-second (Section 8). Global trends are
        only ever considered if no India-scoped trend produced a scored
        match for this exact board, and only when the caller explicitly
        permits it."""
        india_trends = cls.get_active_trends(
            target_date=target_date, gender=gender, canonical_occasion=canonical_occasion,
            target_region=target_region, allow_global_fallback=False,
        )
        scored = [
            r for r in (cls.score_board_against_trend(board_items, t) for t in india_trends) if r
        ]

        if not scored and allow_global_fallback:
            all_trends = cls.get_active_trends(
                target_date=target_date, gender=gender, canonical_occasion=canonical_occasion,
                target_region=target_region, allow_global_fallback=True,
            )
            global_only = [t for t in all_trends if t not in india_trends]
            scored = [
                r for r in (cls.score_board_against_trend(board_items, t) for t in global_only) if r
            ]

        if not scored:
            return None

        scored.sort(key=lambda x: (-x["match_score"], x["trend_id"]))  # deterministic tiebreak
        return scored[0]
