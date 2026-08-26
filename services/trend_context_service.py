import logging
import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional
from services.style_trend_registry import get_trend_registry

MIN_BOARD_TREND_THRESHOLD = 0.55
MIN_TREND_CONFIDENCE = 0.60
logger = logging.getLogger("ahvi.services.trend_context")

_CANONICAL_OCCASION_MAP = {
    "office": "office", "work": "office", "workwear": "workwear", "business": "office",
    "corporate": "office", "meeting": "office", "presentation": "office",
    "dinner": "dinner", "date": "dinner", "date_night": "dinner", "evening": "evening",
    "casual": "casual", "daily": "casual", "everyday": "casual", "lounge": "casual",
    "smart_casual": "smart_casual", "polished": "smart_casual",
    "weekend": "weekend", "brunch": "brunch", "outing": "weekend",
    "wedding": "wedding", "sangeet": "wedding", "reception": "wedding", "marriage": "wedding",
    "festive": "festive", "diwali": "festive", "eid": "festive", "holi": "festive",
    "party": "party", "club": "party", "cocktail": "party", "night_out": "party",
    "travel": "travel", "vacation": "travel", "flight": "travel", "airport": "travel",
    "workout": "casual", "gym": "casual", "sport": "casual", "beach": "travel", "beach_party": "party",
}


def _canonicalize_occasion(occasion: Optional[str]) -> Optional[str]:
    if not occasion:
        return None
    raw = str(occasion).lower().strip().replace(" ", "_").replace("-", "_")
    if not raw:
        return None
    if raw in _CANONICAL_OCCASION_MAP:
        return _CANONICAL_OCCASION_MAP[raw]
    sorted_keys = sorted(_CANONICAL_OCCASION_MAP.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if re.search(r'\b' + re.escape(key) + r'\b', raw) or key in raw:
            return _CANONICAL_OCCASION_MAP[key]
    return None


def annotate_board_with_trend(
    board_dict: dict,
    user_gender: Optional[str] = None,
    occasion: Optional[str] = None,
    target_region: str = "india",
    allow_global_fallback: bool = False,
    target_date: Any = None,
) -> dict:
    """
    Additive soft trend annotation. Modifies board ONLY if a valid, current, source-backed trend matches.
    Trends NEVER alter outfit selection, item order, or styling safety.
    If no match or error occurs, trend keys are cleared and the board is returned intact.
    """
    if not isinstance(board_dict, dict):
        return board_dict
    try:
        items = board_dict.get("items") or board_dict.get("garments") or []
        trend_match = TrendContextService.match_board_trend(
            board_items=items,
            gender=user_gender,
            occasion=occasion,
            target_region=target_region,
            allow_global_fallback=allow_global_fallback,
            target_date=target_date,
        )
        if trend_match:
            board_dict["trend_label"] = trend_match["label"]
            board_dict["trend_explanation"] = trend_match.get("explanation", "")
            board_dict["trend_meta"] = {
                "trend_id": trend_match["trend_id"],
                "match_score": trend_match["match_score"],
                "region": trend_match.get("region", target_region),
                "valid_until": trend_match.get("valid_until"),
                "source_count": trend_match.get("source_count", 1),
                "confidence": trend_match.get("confidence", 0.85),
                "publisher": trend_match.get("publisher", ""),
                "published_at": trend_match.get("published_at", ""),
            }
        else:
            board_dict.pop("trend_label", None)
            board_dict.pop("trend_explanation", None)
            board_dict.pop("trend_meta", None)
    except Exception as exc:
        logger.warning("trend_annotation.failed occasion=%s err=%s", occasion, exc, exc_info=False)

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
                pass
        return None

    @classmethod
    def is_trend_valid(
        cls,
        trend: Dict[str, Any],
        target_date: Any = None,
        target_region: str = "india",
        allow_global_fallback: bool = False,
    ) -> bool:
        """
        Strict freshness, provenance, confidence, review state, and region gate.
        Returns False if expired, unsourced, unpublished, low confidence, or unapproved.
        """
        if not isinstance(trend, dict):
            return False

        # 1. Freshness Gate (strict date parsing — no unsourced 2099 infinity)
        v_from = cls._parse_date(trend.get("valid_from"))
        v_until = cls._parse_date(trend.get("valid_until"))

        if not v_from or not v_until:
            return False

        current_d = cls._parse_date(target_date) or datetime.utcnow().date()
        if not (v_from <= current_d <= v_until):
            return False

        # 2. Source & Provenance Gate
        source = trend.get("source")
        if not isinstance(source, dict):
            return False
        publisher = str(source.get("publisher") or "").strip()
        url = str(source.get("url") or "").strip()
        pub_at = cls._parse_date(source.get("published_at"))

        if not publisher or not url or not pub_at:
            return False

        # 3. Confidence Gate
        try:
            conf = float(trend.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if conf < MIN_TREND_CONFIDENCE:
            return False

        # 4. Review State Gate
        if str(trend.get("review_state") or "approved").lower().strip() != "approved":
            return False

        # 5. Region & Scope Gate
        req_region = str(target_region or "india").lower().strip()
        trend_regions = [str(r).lower().strip() for r in (trend.get("region") or [])]
        trend_scope = str(trend.get("scope") or "").lower().strip()

        if req_region == "india":
            is_india_trend = "india" in trend_regions or trend_scope == "india"
            if not is_india_trend:
                is_global_trend = "global" in trend_regions or trend_scope == "global"
                if not (allow_global_fallback and is_global_trend):
                    return False
        elif req_region:
            if req_region not in trend_regions and trend_scope != req_region:
                if not (allow_global_fallback and ("global" in trend_regions or trend_scope == "global")):
                    return False

        return True

    @classmethod
    def get_active_trends(
        cls,
        target_date: Any = None,
        gender: Optional[str] = None,
        occasion: Optional[str] = None,
        target_region: str = "india",
        allow_global_fallback: bool = False,
    ) -> List[Dict[str, Any]]:
        """Filter registry by strict provenance/freshness gates, gender, and occasion."""
        all_trends = get_trend_registry()
        active = []
        canonical_occ = _canonicalize_occasion(occasion)

        for trend in all_trends:
            if not cls.is_trend_valid(
                trend,
                target_date=target_date,
                target_region=target_region,
                allow_global_fallback=allow_global_fallback,
            ):
                continue

            # Gender filter
            if gender:
                g_norm = gender.lower().strip()
                t_genders = [g.lower() for g in trend.get("gender", [])]
                if g_norm not in t_genders and "unisex" not in t_genders:
                    continue

            # Canonical Occasion filter
            if canonical_occ:
                t_occs = [o.lower().strip().replace(" ", "_") for o in trend.get("occasions", [])]
                if canonical_occ not in t_occs:
                    continue

            active.append(trend)

        return active

    @classmethod
    def get_diagnostics(
        cls,
        target_date: Any = None,
        target_region: str = "india",
    ) -> Dict[str, Any]:
        """Return diagnostic health metrics for monitoring and telemetry."""
        all_trends = get_trend_registry()
        current_d = cls._parse_date(target_date) or datetime.utcnow().date()
        expired_count = 0
        unsourced_count = 0
        active_count = 0
        latest_ingested = ""

        for t in all_trends:
            ing = str(t.get("ingested_at") or "").strip()
            if ing > latest_ingested:
                latest_ingested = ing

            v_until = cls._parse_date(t.get("valid_until"))
            if v_until and v_until < current_d:
                expired_count += 1

            source = t.get("source")
            if not isinstance(source, dict) or not source.get("publisher") or not source.get("url"):
                unsourced_count += 1

            if cls.is_trend_valid(t, target_date=target_date, target_region=target_region):
                active_count += 1

        return {
            "trend_registry_loaded": len(all_trends) > 0,
            "total_record_count": len(all_trends),
            "active_record_count": active_count,
            "latest_ingestion": latest_ingested or "none",
            "expired_count": expired_count,
            "rejected_unsourced_count": unsourced_count,
            "current_region": target_region,
            "source_health": "healthy" if active_count > 0 else "degraded",
        }

    @classmethod
    def score_board_against_trend(
        cls,
        board_items: List[Dict[str, Any]],
        trend: Dict[str, Any],
        target_region: str = "india",
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates items individually.
        An item MUST have a strong signal (keyword/tag match) to qualify.
        Color matching alone cannot produce a false-positive match!
        """
        if not board_items:
            return None

        trend_categories = [c.lower() for c in trend.get("categories", [])]
        trend_colors = [c.lower() for c in trend.get("colors", [])]
        trend_keywords = [k.lower() for k in trend.get("keywords", [])]

        matched_item_count = 0
        matched_item_ids = []
        matched_item_names = []
        matched_reasons = set()
        total_item_points = 0.0

        for item in board_items:
            item_id = str(item.get("id") or item.get("item_id") or item.get("garment_id") or item.get("$id") or "")
            item_name = str(item.get("name") or item.get("title") or item.get("label") or "").strip()
            item_name_lower = item_name.lower()
            item_cat = str(item.get("category") or item.get("main_category") or "").lower()
            item_subcat = str(item.get("sub_category") or item.get("subcategory") or item.get("type") or "").lower()
            item_color = str(item.get("color") or item.get("colour") or "").lower()
            raw_tags = item.get("tags") or item.get("style_tags") or []
            item_tags = [str(t).lower() for t in (raw_tags if isinstance(raw_tags, list) else [])]

            item_text = f"{item_name_lower} {item_subcat} {' '.join(item_tags)}".lower()

            matched_keywords = [kw for kw in trend_keywords if kw in item_text]
            has_strong_signal = len(matched_keywords) > 0

            # Color-only coincidence WITHOUT strong keyword signal is REJECTED
            if not has_strong_signal:
                continue

            item_score = 2.0  # Base points for strong keyword/tag signal

            if any(c in item_color or c in item_name_lower for c in trend_colors):
                item_score += 1.0

            if any(tc in item_cat for tc in trend_categories):
                item_score += 0.5

            total_item_points += item_score
            matched_reasons.update(matched_keywords)
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

        # Personal explanation linking back to matching wardrobe items
        label = str(trend.get("label") or "")
        unique_names = list(dict.fromkeys(matched_item_names))
        if len(unique_names) == 1:
            matched_phrase = unique_names[0].lower()
        elif len(unique_names) >= 2:
            matched_phrase = f"{', '.join(x.lower() for x in unique_names[:-1])} & {unique_names[-1].lower()}"
        else:
            matched_phrase = "existing pieces"

        explanation = f"This {label.lower()} direction is current, and your {matched_phrase} already fit it."

        source = trend.get("source") or {}

        return {
            "trend_id": trend.get("trend_id"),
            "label": label,
            "explanation": explanation,
            "match_score": round(normalized_score, 2),
            "region": target_region,
            "valid_until": trend.get("valid_until"),
            "source_count": 1,
            "confidence": float(trend.get("confidence", 0.85)),
            "publisher": str(source.get("publisher") or ""),
            "published_at": str(source.get("published_at") or ""),
            "matched_item_ids": list(set(matched_item_ids)),
            "matched_reasons": list(matched_reasons)[:3],
        }

    @classmethod
    def match_board_trend(
        cls,
        board_items: List[Dict[str, Any]],
        gender: Optional[str] = None,
        occasion: Optional[str] = None,
        target_region: str = "india",
        allow_global_fallback: bool = False,
        target_date: Any = None,
    ) -> Optional[Dict[str, Any]]:

        active_trends = cls.get_active_trends(
            target_date=target_date,
            gender=gender,
            occasion=occasion,
            target_region=target_region,
            allow_global_fallback=allow_global_fallback,
        )
        if not active_trends:
            return None

        scored_trends = []
        for trend in active_trends:
            res = cls.score_board_against_trend(board_items, trend, target_region=target_region)
            if res:
                scored_trends.append(res)

        if not scored_trends:
            return None

        scored_trends.sort(key=lambda x: x["match_score"], reverse=True)
        return scored_trends[0]
