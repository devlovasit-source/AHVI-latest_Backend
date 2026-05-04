from __future__ import annotations

import json
import logging
import re
import base64
import uuid
from typing import Any, Dict, List

from brain.daily_dependency_engine import build_daily_dependency_response
from brain.engines.style_board_engine import style_board_engine
from brain.engines.style_board_renderer import style_board_renderer
from brain.engines.fitness.fitness_engine import fitness_engine
from brain.engines.proactive_engine import proactive_engine
from brain.intelligence.bank_snippets import (
    color_harmony_snippet,
    weather_overlay_snippet,
    print_pattern_snippet,
    silhouette_snippet,
)
from brain.intent_engine import detect_intent
from brain.outfit_pipeline import get_daily_outfits
from brain.plan_pack_flow import build_plan_pack_response
from brain.response.response_assembler import response_assembler
from brain.tone.tone_engine import tone_engine
from services.appwrite_proxy import AppwriteProxy

logger = logging.getLogger("ahvi.orchestrator")


def _ahvi_orchestrator_merge_card_accessories(cards):
    if not isinstance(cards, list):
        return cards

    for card in cards:
        if not isinstance(card, dict):
            continue

        items = card.get("items")
        if not isinstance(items, list):
            items = []

        accessories = card.get("accessories")
        if not isinstance(accessories, list):
            accessories = []

        seen = {
            str(
                x.get("id")
                or x.get("$id")
                or x.get("item_id")
                or x.get("name")
                or x.get("label")
                or ""
            ).lower()
            for x in items
            if isinstance(x, dict)
        }

        for acc in accessories:
            if not isinstance(acc, dict):
                continue
            key = str(
                acc.get("id")
                or acc.get("$id")
                or acc.get("item_id")
                or acc.get("name")
                or acc.get("label")
                or ""
            ).lower()
            if key and key not in seen:
                items.append(acc)
                seen.add(key)

        card["items"] = items

    return cards


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _first_dict(value: Any) -> Dict[str, Any]:
    """
    Return the first dict from a list-like value; otherwise {}.
    Guards visual-intel/outfit extraction from non-dict list entries.
    """
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return dict(first)
    return {}


def _render_style_boards_for_chat(
    cards: List[Dict[str, Any]],
    context: Dict[str, Any],
    *,
    include_base64: bool,
) -> List[Dict[str, Any]]:
    if not include_base64 or not cards:
        return []

    rendered: List[Dict[str, Any]] = []
    style_dna = _dict(context.get("style_dna"))
    for idx, card in enumerate(cards[:3]):
        if not isinstance(card, dict):
            continue
        items = card.get("items") if isinstance(card.get("items"), list) else []
        if not items:
            continue
        try:
            board = style_board_engine.build_board(
                {"items": items, "score": card.get("score")},
                context,
            )
            image_bytes = style_board_renderer.render(board)
            image_base64 = (
                base64.b64encode(image_bytes).decode("ascii") if image_bytes else None
            )
        except Exception as exc:
            logger.warning("style board render failed: %s", exc)
            board = {}
            image_base64 = None

        rendered.append(
            {
                "board_id": str(uuid.uuid4()),
                "type": "style",
                "label": card.get("title") or "AHVI Style Board",
                "score": card.get("score"),
                "aesthetic": board.get("aesthetic"),
                "vibe": board.get("vibe"),
                "items": items,
                "image_base64": image_base64,
                "image_url": None,
                "board_payload": {
                    "items": items,
                    "aesthetic": board.get("aesthetic"),
                    "vibe": board.get("vibe"),
                    "score": card.get("score"),
                    "style_dna": style_dna,
                    "card_id": card.get("id") or f"outfit_card_{idx + 1}",
                },
            }
        )
    return rendered


def _coerce_wardrobe_payload(value: Any) -> list[dict]:
    """
    Normalize wardrobe payloads coming from UI context and/or Appwrite.
    Prevents silent failures where a dict/string is passed into the outfit pipeline.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
            if isinstance(parsed, dict):
                value = parsed
        except Exception:
            return []

    if isinstance(value, dict):
        for key in ("items", "documents", "wardrobe", "data"):
            inner = value.get(key)
            if isinstance(inner, list):
                return [x for x in inner if isinstance(x, dict)]
        return []

    return []


def _extract_occasion(text: str, slots: Dict[str, Any], context: Dict[str, Any]) -> str:
    explicit = _safe_text(slots.get("occasion") or context.get("occasion"))
    if explicit:
        return explicit.lower()

    lowered = (text or "").lower()
    if "wedding" in lowered:
        return "wedding"
    if "party" in lowered:
        return "party"
    if "office" in lowered or "work" in lowered:
        return "office"
    if "date" in lowered:
        return "date_night"
    if "travel" in lowered or "trip" in lowered:
        return "travel"
    return ""


def _normalize_weather_context(context: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(context)
    weather_data = _dict(out.get("weather_data"))
    if not weather_data and isinstance(out.get("user_profile"), dict):
        weather_data = _dict(_dict(out["user_profile"]).get("weather"))
    out["weather_data"] = weather_data
    if not out.get("weather") and weather_data.get("condition"):
        out["weather"] = _safe_text(weather_data.get("condition")).lower()
    return out


def _resolve_organize_module(
    text: str, slots: Dict[str, Any], context: Dict[str, Any]
) -> str:
    slot_module = _safe_text(slots.get("module")).lower()
    if slot_module:
        return slot_module

    module_context = _safe_text(context.get("module_context")).lower()
    if module_context:
        if (
            "meal" in module_context
            or "diet" in module_context
            or "nutrition" in module_context
        ):
            return "meal_planner"
        if (
            "workout" in module_context
            or "fitness" in module_context
            or "gym" in module_context
        ):
            return "workout"
        if "skin" in module_context:
            return "skincare"
        if "calendar" in module_context:
            return "calendar"
        if "bill" in module_context:
            return "bills"
        if "contact" in module_context:
            return "contacts"
        if "goal" in module_context:
            return "life_goals"
        if "life" in module_context:
            return "life_boards"

    lowered = (text or "").lower()
    if any(k in lowered for k in ["meal", "diet", "nutrition", "calorie", "protein"]):
        return "meal_planner"
    if any(k in lowered for k in ["workout", "fitness", "gym", "exercise", "training"]):
        return "workout"
    if "skin" in lowered:
        return "skincare"
    if "calendar" in lowered:
        return "calendar"
    if "bill" in lowered:
        return "bills"
    if "contact" in lowered:
        return "contacts"
    if "goal" in lowered:
        return "life_goals"
    return "life_boards"


def _extract_duration_minutes(
    text: str, context: Dict[str, Any], default: int = 20
) -> int:
    raw = context.get("duration")
    try:
        if raw:
            return max(5, min(90, int(raw)))
    except Exception:
        pass
    match = re.search(r"\b(\d{1,2})\s*(?:min|minute|minutes)\b", text or "", re.I)
    if match:
        return max(5, min(90, int(match.group(1))))
    return default


def _wardrobe_from_appwrite(user_id: str) -> List[Dict[str, Any]]:
    try:
        docs = AppwriteProxy().list_documents("outfits", user_id=user_id, limit=200)
    except Exception:
        return []
    if not isinstance(docs, list):
        return []
    rows: List[Dict[str, Any]] = []
    for d in docs:
        if not isinstance(d, dict):
            continue
        rows.append(
            {
                "id": d.get("$id") or d.get("id"),
                "name": d.get("name"),
                "image_url": d.get("image_url")
                or d.get("imageUrl")
                or d.get("raw_image_url")
                or d.get("url"),
                "imageUrl": d.get("imageUrl")
                or d.get("image_url")
                or d.get("raw_image_url")
                or d.get("url"),
                "masked_url": d.get("masked_url")
                or d.get("maskedUrl")
                or d.get("sticker_url")
                or d.get("cutout_url")
                or d.get("image_url")
                or d.get("imageUrl")
                or d.get("url"),
                "maskedUrl": d.get("maskedUrl")
                or d.get("masked_url")
                or d.get("sticker_url")
                or d.get("cutout_url")
                or d.get("imageUrl")
                or d.get("image_url")
                or d.get("url"),
                "category": d.get("category") or d.get("main_category"),
                "sub_category": d.get("sub_category") or d.get("subcategory"),
                "color": d.get("color") or d.get("color_code"),
                "pattern": d.get("pattern"),
                "occasion_tags": d.get("occasions") or [],
                "weather_tags": d.get("weather") or [],
                "style": d.get("style") or d.get("vibe"),
                "fabric": d.get("fabric"),
                "fit": d.get("fit"),
            }
        )
    return rows


def _safe_list_documents(
    collection: str, user_id: str, limit: int = 50
) -> List[Dict[str, Any]]:
    try:
        rows = AppwriteProxy().list_documents(collection, user_id=user_id, limit=limit)
    except Exception:
        return []
    return (
        [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    )


def _organize_hub_response(
    *,
    uid: str,
    query: str,
    module_key: str,
    context: Dict[str, Any],
    user_profile: Dict[str, Any],
) -> Dict[str, Any]:
    module = _safe_text(module_key).lower() or "life_boards"

    if module == "meal_planner":
        plans = _safe_list_documents("meal_plans", uid, limit=25)
        latest_name = (
            _safe_text(
                (plans[0] if plans else {}).get("name")
                or (plans[0] if plans else {}).get("title")
            )
            if plans
            else ""
        )
        msg = f"You have {len(plans)} meal plans. I can help you build a diet-focused weekly flow."
        if latest_name:
            msg = f"You have {len(plans)} meal plans. Latest plan: {latest_name}."
        toned = tone_engine.apply(
            msg,
            user_profile=user_profile,
            signals={"context_mode": "planning", **_dict(context.get("signals"))},
            context=context,
        )
        return {
            "success": True,
            "message": toned,
            "board": "meal_planner",
            "type": "cards",
            "cards": [
                {
                    "id": "meal_count",
                    "title": "Meal Plans",
                    "kind": "stat",
                    "value": len(plans),
                },
                {
                    "id": "meal_open",
                    "title": "Open Meal Planner",
                    "kind": "action",
                    "action": {
                        "type": "open_module",
                        "module": "meal_planner",
                        "route": "/organize/meal-planner",
                    },
                },
                {
                    "id": "meal_new",
                    "title": "Create Diet Week",
                    "kind": "action",
                    "action": {
                        "type": "open_module",
                        "module": "meal_planner",
                        "route": "/organize/meal-planner?create=1",
                    },
                },
            ],
            "data": {
                "module": module,
                "total_plans": len(plans),
                "latest_plan": latest_name or None,
            },
        }

    if module == "workout":
        workouts = _safe_list_documents("workout_outfits", uid, limit=25)
        duration = _extract_duration_minutes(query, context)
        fitness_input = {
            "goal": "general_fitness",
            "gender": _safe_text(user_profile.get("gender") or "universal").lower()
            or "universal",
            "duration": duration,
            "location": _safe_text(context.get("location") or "home").lower() or "home",
            "equipment": _safe_text(context.get("equipment") or "none").lower()
            or "none",
        }
        rec = fitness_engine.recommend_workout(fitness_input)
        rec_items = (
            rec.get("recommendations")
            if isinstance(rec, dict) and isinstance(rec.get("recommendations"), list)
            else []
        )
        if (
            not rec_items
            and isinstance(rec, dict)
            and isinstance(rec.get("fallback"), list)
        ):
            rec_items = rec.get("fallback") or []
        rec_items = sorted(
            [x for x in rec_items if isinstance(x, dict)],
            key=lambda r: _safe_text(r.get("title")),
        )[:3]
        msg = f"Here's a {duration}-minute workout plan with warmup, main work, and cooldown."
        toned = tone_engine.apply(
            msg,
            user_profile=user_profile,
            signals={"context_mode": "workout", **_dict(context.get("signals"))},
            context=context,
        )
        cards: List[Dict[str, Any]] = [
            {
                "id": "workout_meta",
                "title": f"{duration} min",
                "kind": "stat",
                "value": duration,
                "label": "Duration",
            },
            {
                "id": "workout_equipment",
                "title": "Equipment",
                "kind": "stat",
                "value": fitness_input["equipment"],
            },
            {
                "id": "workout_open",
                "title": "Open Fitness",
                "kind": "action",
                "action": {
                    "type": "open_module",
                    "module": "workout",
                    "route": "/organize/workout",
                },
            },
        ]
        for idx, row in enumerate(rec_items, start=1):
            source_cards = (
                row.get("cards", []) if isinstance(row.get("cards"), list) else []
            )
            checklist_items: List[str] = []
            for source_card in source_cards:
                if isinstance(source_card, dict):
                    section = _safe_text(source_card.get("title"))
                    for item in source_card.get("items", []) or []:
                        checklist_items.append(
                            f"{section}: {item}" if section else str(item)
                        )
            if not checklist_items:
                checklist_items = [
                    "Warmup: 3 minutes easy mobility",
                    "Main circuit: 12 minutes steady effort",
                    "Cooldown: 5 minutes stretch and breathing",
                ]
            cards.append(
                {
                    "id": f"workout_rec_{idx}",
                    "title": _safe_text(row.get("title")) or f"Workout {idx}",
                    "kind": "checklist",
                    "items": checklist_items[:8],
                    "duration": row.get("duration_min") or duration,
                    "equipment": row.get("equipment") or fitness_input["equipment"],
                }
            )
        return {
            "success": True,
            "message": toned,
            "board": "fitness",
            "type": "workout_plan",
            "cards": cards,
            "data": {
                "module": "fitness",
                "duration": duration,
                "equipment": fitness_input["equipment"],
                "saved_workouts": len(workouts),
                "recommendations": rec_items,
            },
        }

    route_map = {
        "skincare": ("/organize/skincare", "skincare"),
        "calendar": ("/organize/calendar", "calendar"),
        "bills": ("/organize/bills", "bills"),
        "contacts": ("/organize/contacts", "contacts"),
        "life_goals": ("/organize/life-goals", "life_goals"),
        "life_boards": ("/organize/life-boards", "life_boards"),
        "medicines": ("/organize/medicines", "medicines"),
    }
    route, board = route_map.get(module, ("/organize/life-boards", "organize"))
    msg = f"I routed this to {module.replace('_', ' ')} so you can continue in the right board."
    toned = tone_engine.apply(
        msg,
        user_profile=user_profile,
        signals={"context_mode": "planning", **_dict(context.get("signals"))},
        context=context,
    )
    return {
        "success": True,
        "message": toned,
        "board": board,
        "type": "cards",
        "cards": [
            {
                "id": "organize_open",
                "title": f"Open {module.replace('_', ' ').title()}",
                "kind": "action",
                "action": {"type": "open_module", "module": module, "route": route},
            }
        ],
        "data": {"module": module},
    }


def _visual_intelligence_from_outfit(outfit: Dict[str, Any]) -> Dict[str, Any]:
    parts = [
        _dict(outfit.get("top")),
        _dict(outfit.get("bottom")),
        _dict(outfit.get("dress")),
        _dict(outfit.get("shoes")),
    ] + [x for x in (outfit.get("accessories") or []) if isinstance(x, dict)]
    colors = [
        _safe_text(p.get("color")).lower() for p in parts if _safe_text(p.get("color"))
    ]
    patterns = [
        _safe_text(p.get("pattern")).lower()
        for p in parts
        if _safe_text(p.get("pattern"))
    ]
    styles = [
        _safe_text(p.get("style")).lower() for p in parts if _safe_text(p.get("style"))
    ]
    return {
        "dominant_palette": sorted(set(colors))[:4],
        "pattern_mix": sorted(set(patterns))[:4],
        "style_signals": sorted(set(styles))[:4],
        "composition_score": float(outfit.get("score") or 0.0),
        "story": _safe_text(
            _dict(outfit.get("story")).get("subtitle") or outfit.get("explanation")
        ),
    }


class AhviOrchestrator:
    def run(
        self,
        *,
        text: str,
        user_id: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        query = _safe_text(text)
        uid = (
            _safe_text(user_id) or _safe_text(_dict(context).get("user_id")) or "user_1"
        )
        ctx = _normalize_weather_context(_dict(context))
        ctx["user_id"] = uid  # used by memory scoring + downstream engines
        user_profile = _dict(ctx.get("user_profile"))

        # Proactive signals: enrich "signals" deterministically so downstream engines can use them.
        try:
            ctx = proactive_engine.inject(ctx)
        except Exception as exc:
            logger.warning("proactive_engine.inject failed: %s", str(exc))
        signals = _dict(ctx.get("signals"))
        proactive_signals = _dict(ctx.get("proactive_signals"))
        ctx["signals"] = {**signals, **proactive_signals}

        intent_row = detect_intent(query, history=(ctx.get("history") or []))
        intent = _safe_text(intent_row.get("intent")).lower() or "general"
        slots = _dict(intent_row.get("slots"))
        occasion = _extract_occasion(query, slots, ctx)
        if occasion:
            ctx["occasion"] = occasion

        if intent == "daily_dependency":
            out = build_daily_dependency_response(user_id=uid, context=ctx)
            out["message"] = tone_engine.apply(
                _safe_text(out.get("message")),
                user_profile=user_profile,
                signals={"context_mode": "planning", **_dict(ctx.get("signals"))},
                context=ctx,
            )
            out["meta"] = {
                **_dict(out.get("meta")),
                "intent": intent,
                "confidence": float(intent_row.get("confidence", 0.0)),
            }
            return out

        if intent == "plan_pack":
            out = build_plan_pack_response(query, ctx)
            out["success"] = True
            out["message"] = tone_engine.apply(
                _safe_text(out.get("message")),
                user_profile=user_profile,
                signals={"context_mode": "travel", **_dict(ctx.get("signals"))},
                context=ctx,
            )
            out["meta"] = {
                **_dict(out.get("meta")),
                "intent": intent,
                "confidence": float(intent_row.get("confidence", 0.0)),
            }
            return out

        if intent == "wardrobe_query":
            docs = _wardrobe_from_appwrite(uid)
            toned = tone_engine.apply(
                f"You currently have {len(docs)} items in your wardrobe.",
                user_profile=user_profile,
                signals={"context_mode": "home", **_dict(ctx.get("signals"))},
                context=ctx,
            )
            return {
                "success": True,
                "message": toned,
                "board": "wardrobe",
                "type": "stats",
                "cards": [],
                "data": {"total_items": len(docs)},
                "meta": {
                    "intent": intent,
                    "confidence": float(intent_row.get("confidence", 0.0)),
                },
            }

        if intent == "organize_hub":
            module_key = _resolve_organize_module(query, slots, ctx)
            out = _organize_hub_response(
                uid=uid,
                query=query,
                module_key=module_key,
                context=ctx,
                user_profile=user_profile,
            )
            out["meta"] = {
                **_dict(out.get("meta")),
                "intent": intent,
                "module": module_key,
                "confidence": float(intent_row.get("confidence", 0.0)),
            }
            return out

        module_key = _resolve_organize_module(query, slots, ctx)
        if module_key == "workout":
            out = _organize_hub_response(
                uid=uid,
                query=query,
                module_key=module_key,
                context=ctx,
                user_profile=user_profile,
            )
            out["meta"] = {
                **_dict(out.get("meta")),
                "intent": intent,
                "module": module_key,
                "confidence": float(intent_row.get("confidence", 0.0)),
            }
            return out

        # If the user mentioned a clothing-relevant occasion (date night,
        # office, party, vacation, brunch, etc.) we treat the request as an
        # outfit query even when the intent classifier didn't tag it as one.
        # Without this, "I have a date tonight — what should I wear?" falls
        # through to the generic fallback and the chat returns no cards.
        if (
            intent not in {"daily_outfit", "occasion_outfit", "explore_styles"}
            and occasion
        ):
            intent = "occasion_outfit"

        if intent in {"daily_outfit", "occasion_outfit", "explore_styles"}:
            wardrobe_ctx = ctx.get("wardrobe")
            wardrobe = _coerce_wardrobe_payload(wardrobe_ctx)
            wardrobe_source = "ctx" if wardrobe else "appwrite"
            if not wardrobe:
                wardrobe = _wardrobe_from_appwrite(uid)

            try:
                outfit_result = get_daily_outfits(
                    {
                        "user_id": uid,
                        "wardrobe": wardrobe,
                        "context": {
                            **ctx,
                            "query": query,
                            "occasion": occasion or _safe_text(ctx.get("occasion")),
                        },
                    }
                )
            except Exception as exc:
                logger.exception(
                    "get_daily_outfits failed uid=%s intent=%s error=%s",
                    uid,
                    intent,
                    str(exc),
                )
                toned = tone_engine.apply(
                    "I hit a temporary issue while generating outfits. Please try again in a moment.",
                    user_profile=user_profile,
                    signals={"context_mode": "styling", **_dict(ctx.get("signals"))},
                    context=ctx,
                )
                return {
                    "success": True,
                    "message": toned,
                    "board": "style",
                    "type": "text",
                    "cards": [],
                    "data": {},
                    "meta": {
                        "intent": intent,
                        "confidence": float(intent_row.get("confidence", 0.0)),
                        "wardrobe_source": wardrobe_source,
                        "wardrobe_count": len(wardrobe),
                        "error": "outfit_pipeline_failed",
                    },
                }

            outfits = (
                outfit_result.get("outfits")
                if isinstance(outfit_result.get("outfits"), list)
                else []
            )
            visual_intel = (
                _visual_intelligence_from_outfit(_first_dict(outfits))
                if outfits
                else {}
            )

            message = _safe_text(
                outfit_result.get("context") or outfit_result.get("message")
            )
            if outfits and not message:
                message = _safe_text(
                    _dict(_dict(outfits[0]).get("story")).get("subtitle")
                    or _dict(outfits[0]).get("explanation")
                )
            if not message:
                message = "Your visual intelligence result is ready."

            first_outfit = _dict(outfits[0]) if outfits else {}
            # Make the surface layer reflect the real intelligence:
            # pull deterministic bank snippets + scorer reasons into the message.
            try:
                board_item_ids_for_key = (
                    outfit_result.get("board_item_ids")
                    if isinstance(outfit_result.get("board_item_ids"), list)
                    else []
                )
                stable_key = "|".join(
                    [str(x).strip() for x in board_item_ids_for_key if str(x).strip()]
                ) or str(first_outfit.get("combo_id") or "outfit")

                breakdown = _dict(first_outfit.get("score_breakdown"))
                color_hint = float(breakdown.get("color_intelligence") or 0.0)
                silhouette_hint = float(breakdown.get("style_graph") or 0.0)
                weather_mode = _safe_text(
                    _dict(ctx.get("signals")).get("weather_mode")
                    or ctx.get("weather")
                    or ""
                )

                harmony_line = color_harmony_snippet(color_hint, key=stable_key)
                weather_line = (
                    weather_overlay_snippet(weather_mode, key=stable_key)
                    if weather_mode
                    else ""
                )

                # Print + silhouette banks
                items_for_patterns = (
                    first_outfit.get("refined_items")
                    if isinstance(first_outfit.get("refined_items"), list)
                    else first_outfit.get("items")
                )
                patterns = []
                if isinstance(items_for_patterns, list):
                    patterns = [
                        str(_dict(x).get("pattern") or "").strip().lower()
                        for x in items_for_patterns
                        if _dict(x)
                    ]
                print_line = print_pattern_snippet(patterns, key=stable_key)
                silhouette_line = silhouette_snippet(silhouette_hint, key=stable_key)

                unified = _dict(first_outfit.get("unified_style"))
                reasons = (
                    unified.get("reasons")
                    if isinstance(unified.get("reasons"), list)
                    else []
                )
                reason_line = ""
                if reasons:
                    reason_line = (
                        "Why it works: "
                        + ", ".join(
                            [_safe_text(r) for r in reasons if _safe_text(r)][:2]
                        )
                        + "."
                    )

                extra_lines = " ".join(
                    [
                        x
                        for x in [
                            harmony_line,
                            print_line,
                            silhouette_line,
                            weather_line,
                            reason_line,
                        ]
                        if x
                    ]
                )
                if extra_lines:
                    message = (message.rstrip(".") + ". " + extra_lines).strip()
            except Exception:
                pass
            toned = tone_engine.apply(
                message,
                user_profile=user_profile,
                signals={"context_mode": "styling", **_dict(ctx.get("signals"))},
                context={
                    "aesthetic": first_outfit.get("aesthetic"),
                    "outfit_data": {
                        "items": [
                            x for x in first_outfit.values() if isinstance(x, dict)
                        ]
                    },
                },
            )

            # ================= AHVI CLEAN STYLE RETURN V1 BEGIN =================
            board_item_ids = (
                outfit_result.get("board_item_ids")
                if isinstance(outfit_result.get("board_item_ids"), list)
                else []
            )
            board_item_ids = [str(x).strip() for x in board_item_ids if str(x).strip()]
            primary_board_id = board_item_ids[0] if board_item_ids else ""

            cards = (
                outfit_result.get("cards")
                if isinstance(outfit_result.get("cards"), list)
                else []
            )
            cards = _ahvi_orchestrator_merge_card_accessories(cards)

            try:
                logger.info(
                    "style card detail uid=%s card_item_counts=%s accessory_counts=%s first_card_items=%s",
                    uid,
                    [len(c.get("items") or []) for c in cards if isinstance(c, dict)],
                    [
                        len(c.get("accessories") or [])
                        for c in cards
                        if isinstance(c, dict)
                    ],
                    [
                        str((i or {}).get("name") or (i or {}).get("label") or "")
                        for i in (
                            (
                                cards[0].get("items")
                                if cards and isinstance(cards[0], dict)
                                else []
                            )
                            or []
                        )
                        if isinstance(i, dict)
                    ][:8],
                )
            except Exception:
                pass

            rendered_boards = _render_style_boards_for_chat(
                cards,
                {
                    **ctx,
                    "query": query,
                    "occasion": occasion or _safe_text(ctx.get("occasion")),
                },
                include_base64=bool(ctx.get("include_base64")),
            )

            try:
                logger.info(
                    "style intent=%s uid=%s wardrobe_source=%s wardrobe_count=%s cards=%s board_id=%s",
                    intent,
                    uid,
                    wardrobe_source,
                    len(wardrobe),
                    len(cards),
                    primary_board_id,
                )
            except Exception:
                pass

            return {
                "success": True,
                "message": toned,
                "board": "style",
                "type": "cards",
                "cards": cards,
                "board_ids": primary_board_id,
                "data": {
                    "outfits": outfits,
                    "visual_intelligence": visual_intel,
                    "pipeline": _dict(outfit_result.get("pipeline")),
                    "rendered_boards": rendered_boards,
                    "board_item_ids": board_item_ids,
                },
                "meta": {
                    "intent": intent,
                    "confidence": float(intent_row.get("confidence", 0.0)),
                    "visual_intelligence_enabled": True,
                    "wardrobe_source": wardrobe_source,
                    "wardrobe_count": len(wardrobe),
                },
            }
            # ================= AHVI CLEAN STYLE RETURN V1 END =================

        merged = {
            "type": "general",
            "message": "Tell me what you need: outfit, planning, or organizing help.",
            "data": {},
        }
        fallback = response_assembler.assemble(
            merged_output=merged,
            context={"user_profile": user_profile, "signals": {"context_mode": "home"}},
        )
        return {
            "success": True,
            "message": fallback,
            "board": "general",
            "type": "text",
            "cards": [],
            "data": {},
            "meta": {
                "intent": intent,
                "confidence": float(intent_row.get("confidence", 0.0)),
            },
        }


ahvi_orchestrator = AhviOrchestrator()
orchestrator = ahvi_orchestrator

# ================= AHVI ORCHESTRATOR STYLE DNA PATCH V2 BEGIN =================

try:
    _AHVI_ORIGINAL_ORCHESTRATOR_RUN = AhviOrchestrator.run
except Exception:
    _AHVI_ORIGINAL_ORCHESTRATOR_RUN = None


if _AHVI_ORIGINAL_ORCHESTRATOR_RUN and not getattr(
    AhviOrchestrator.run, "_ahvi_style_dna_preflight_v2", False
):

    def _ahvi_orchestrator_run_with_style_dna(self, text, user_id=None, context=None):
        ctx = dict(context or {})

        if user_id and not ctx.get("user_id"):
            ctx["user_id"] = user_id

        try:
            from services.data_access_service import (
                get_user_profile,
                merge_user_profiles,
            )

            stored_profile = get_user_profile(
                user_id=str(user_id or ctx.get("user_id") or "").strip()
            )
            request_profile = (
                ctx.get("user_profile")
                if isinstance(ctx.get("user_profile"), dict)
                else {}
            )
            ctx["user_profile"] = merge_user_profiles(stored_profile, request_profile)
        except Exception as exc:
            try:
                logger.warning(
                    "user profile merge failed uid=%s error=%s", user_id, str(exc)
                )
            except Exception:
                pass

        try:
            from brain.personalization.style_dna_engine import style_dna_engine

            enriched = style_dna_engine.enrich_context(ctx)
            if isinstance(enriched, dict):
                ctx = enriched
        except Exception as exc:
            try:
                logger.warning(
                    "style_dna preflight failed uid=%s error=%s", user_id, str(exc)
                )
            except Exception:
                pass

        return _AHVI_ORIGINAL_ORCHESTRATOR_RUN(
            self,
            text=text,
            user_id=user_id,
            context=ctx,
        )

    _ahvi_orchestrator_run_with_style_dna._ahvi_style_dna_preflight_v2 = True
    AhviOrchestrator.run = _ahvi_orchestrator_run_with_style_dna

# ================= AHVI ORCHESTRATOR STYLE DNA PATCH V2 END =================
