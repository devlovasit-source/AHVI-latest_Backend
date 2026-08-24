"""
services/trend_context_service.py
Deterministic matching and scoring service for AHVI MVP Trend Intelligence V1.
"""
from datetime import datetime, date
from typing import Any, Dict, List, Optional
from services.style_trend_registry import get_trend_registry

MIN_BOARD_TREND_THRESHOLD = 0.55


class TrendContextService:
    @staticmethod
    def _parse_date(target_date: Any) -> date:
        if isinstance(target_date, datetime):
            return target_date.date()
        if isinstance(target_date, date):
            return target_date
        if isinstance(target_date, str):
            try:
                return datetime.fromisoformat(target_date.split("T")[0]).date()
            except Exception:
                pass
        return datetime.utcnow().date()

    @classmethod
    def get_active_trends(
        cls,
        target_date: Any = None,
        gender: Optional[str] = None,
        occasion: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Filter registry by date, gender, and strict occasion."""
        current_d = cls._parse_date(target_date)
        all_trends = get_trend_registry()
        active = []

        for trend in all_trends:
            v_from = cls._parse_date(trend.get("valid_from", "1970-01-01"))
            v_until = cls._parse_date(trend.get("valid_until", "2099-12-31"))

            # 1. Date filter
            if not (v_from <= current_d <= v_until):
                continue

            # 2. Gender filter
            if gender:
                g_norm = gender.lower().strip()
                t_genders = [g.lower() for g in trend.get("gender", [])]
                if g_norm not in t_genders and "unisex" not in t_genders:
                    continue

            # 3. Strict Occasion filter (NO fallback for 'casual')
            if occasion:
                occ_norm = occasion.lower().strip().replace(" ", "_")
                t_occs = [o.lower().strip().replace(" ", "_") for o in trend.get("occasions", [])]
                if occ_norm not in t_occs:
                    continue

            active.append(trend)

        return active

    @classmethod
    def score_board_against_trend(
        cls,
        board_items: List[Dict[str, Any]],
        trend: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates score by evaluating items individually.
        An item MUST have a strong signal (keyword/tag match) to qualify.
        """
        if not board_items:
            return None

        trend_categories = [c.lower() for c in trend.get("categories", [])]
        trend_colors = [c.lower() for c in trend.get("colors", [])]
        trend_keywords = [k.lower() for k in trend.get("keywords", [])]

        matched_item_ids = []
        matched_reasons = set()
        total_item_points = 0.0

        for item in board_items:
            item_id = str(item.get("id") or item.get("item_id") or item.get("garment_id") or item.get("$id") or "")
            item_name = str(item.get("name") or item.get("title") or item.get("label") or "").lower()
            item_cat = str(item.get("category") or item.get("main_category") or "").lower()
            item_subcat = str(item.get("sub_category") or item.get("subcategory") or item.get("type") or "").lower()
            item_color = str(item.get("color") or item.get("colour") or "").lower()
            raw_tags = item.get("tags") or item.get("style_tags") or []
            item_tags = [str(t).lower() for t in (raw_tags if isinstance(raw_tags, list) else [])]
            
            # Construct a searchable string for keywords
            item_text = f"{item_name} {item_subcat} {' '.join(item_tags)}".lower()


            # STRONG SIGNAL CHECK (Must match keyword, tag, or subcategory)
            matched_keywords = [kw for kw in trend_keywords if kw in item_text]
            has_strong_signal = len(matched_keywords) > 0

            if not has_strong_signal:
                continue

            # It's a match. Calculate item points.
            item_score = 2.0  # Base points for strong signal
            
            # Supporting signal: Color (+1.0)
            if any(c in item_color or c in item_name for c in trend_colors):
                item_score += 1.0
                
            # Supporting signal: Broad Category (+0.5)
            if any(tc in item_cat for tc in trend_categories):
                item_score += 0.5

            total_item_points += item_score
            matched_reasons.update(matched_keywords)
            if item_id:
                matched_item_ids.append(item_id)

        # Board Aggregation
        if not matched_item_ids:
            return None

        # Max possible points per matched item is 3.5
        avg_item_score = total_item_points / len(matched_item_ids)
        coverage = len(matched_item_ids) / len(board_items)
        
        # Final score relies heavily on coverage (how much of the outfit fits the trend)
        normalized_score = (avg_item_score / 3.5) * coverage
        
        # CONSERVATIVE FIX: Minor boost if coverage is exceptionally high (e.g., >= 50%)
        if coverage >= 0.5:
            normalized_score = min(normalized_score + 0.1, 1.0)

        if normalized_score < MIN_BOARD_TREND_THRESHOLD:
            return None

        return {
            "trend_id": trend.get("trend_id"),
            "label": trend.get("label"),
            "match_score": round(normalized_score, 2),
            "matched_item_ids": list(set(matched_item_ids)),
            "matched_reasons": list(matched_reasons)[:3]
        }

    @classmethod
    def match_board_trend(
        cls,
        board_items: List[Dict[str, Any]],
        gender: Optional[str] = None,
        occasion: Optional[str] = None,
        target_date: Any = None
    ) -> Optional[Dict[str, Any]]:
        
        active_trends = cls.get_active_trends(target_date=target_date, gender=gender, occasion=occasion)
        if not active_trends:
            return None

        scored_trends = []
        for trend in active_trends:
            res = cls.score_board_against_trend(board_items, trend)
            if res:
                scored_trends.append(res)

        if not scored_trends:
            return None

        # Sort by match_score descending
        scored_trends.sort(key=lambda x: x["match_score"], reverse=True)
        return scored_trends[0]
