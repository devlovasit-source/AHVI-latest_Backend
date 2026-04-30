
import logging
import os
from typing import Dict, Any, List

from brain.tone.tone_engine import tone_engine
from services.llm_service import (
    generate_outfit_explanation,
    generate_style_advice,
    generate_followup_suggestions,
)
from brain.intelligence.bank_snippets import (
    color_harmony_snippet,
    weather_overlay_snippet,
    print_pattern_snippet,
    silhouette_snippet,
)

logger = logging.getLogger("ahvi.response_assembler")


class ResponseAssembler:

    # =========================
    # MAIN ENTRY
    # =========================
    def assemble(self, merged_output: dict, context: dict = None) -> dict:
        context = context or {}
        response_type = merged_output.get("type")

        if response_type == "styling":
            return self._assemble_styling(merged_output, context)

        return self._assemble_multi_domain(merged_output, context)

    # =========================
    # 🔥 STYLING (ELITE)
    # =========================
    def _assemble_styling(self, merged_output: dict, context: dict) -> dict:

        data = merged_output.get("data", {})
        outfits = data.get("outfits", [])
        boards = data.get("boards", [])

        if not outfits:
            return self._wrap_response(
                "I need a bit more to work with — want me to refine it?",
                context,
                chips=["Show outfit ideas", "Try different vibe"]
            )

        user_profile = context.get("user_profile", {})
        signals = context.get("signals", {})

        # -------------------------
        # BASE VISUAL TEXT (deterministic + intelligence-backed)
        # -------------------------
        base_text = self._build_visual_explanation(outfits[0], context)
        bank_text = self._bank_intelligence(outfits[0], context)
        if bank_text:
            base_text = f"{base_text} {bank_text}".strip()

        # -------------------------
        # LLM ENHANCEMENT
        # -------------------------
        if self._llm_enabled():
            try:
                llm_text = generate_outfit_explanation(
                    outfits=outfits,
                    context=context,
                    user_profile=user_profile,
                    signals=signals,
                )

                if isinstance(llm_text, dict):
                    llm_text = llm_text.get("content", "")

                if llm_text:
                    base_text = llm_text

            except Exception as e:
                logger.warning("LLM styling failed: %s", e)

        # -------------------------
        # FINAL MESSAGE (PERSONALITY RESTORED)
        # -------------------------
        message = self._compose([
            self._proactive_prefix(signals),
            self._reaction(outfits[0]),
            base_text,
            self._closer()
        ])

        message = tone_engine.apply(
            message,
            user_profile=user_profile,
            signals=signals,
        )

        # -------------------------
        # CHIPS
        # -------------------------
        chips = self._safe_actions(context)

        return self._wrap_response(
            message,
            context,
            chips=chips,
            cards=boards,
            data={"outfits": outfits}
        )

    # =========================
    # 🧠 MULTI DOMAIN
    # =========================
    def _assemble_multi_domain(self, merged_output: dict, context: dict) -> dict:

        data = merged_output.get("data", {})
        user_profile = context.get("user_profile", {})
        signals = context.get("signals", {})

        base_text = merged_output.get("message") or self._fallback(data)

        if self._llm_enabled():
            try:
                improved = generate_style_advice(
                    user_input=base_text,
                    wardrobe_summary=str(context.get("wardrobe", "")),
                    user_profile=user_profile,
                    signals=signals,
                )
                if improved:
                    base_text = improved
            except Exception as e:
                logger.warning("LLM multi-domain failed: %s", e)

        message = self._compose([
            self._reaction(),
            base_text,
            self._closer()
        ])

        message = tone_engine.apply(
            message,
            user_profile=user_profile,
            signals=signals,
        )

        return self._wrap_response(
            message,
            context,
            chips=self._safe_actions(context),
            data=data
        )

    # =========================
    # 🔥 RESPONSE WRAPPER (RESTORED CLEANLY)
    # =========================
    def _wrap_response(
        self,
        text: str,
        context: dict,
        chips: List[str] = None,
        cards: List[dict] = None,
        data: Dict[str, Any] = None
    ) -> dict:

        return {
            "message": {
                "role": "assistant",
                "content": text.strip()
            },
            "chips": chips or [],
            "cards": cards or [],
            "data": data or {},
            "meta": context.get("intent_meta", {})
        }

    # =========================
    # 🧠 TEXT COMPOSER (LIMIT RESTORED)
    # =========================
    def _compose(self, parts: List[str]) -> str:
        parts = [p for p in parts if p]
        text = " ".join(parts)

        sentences = text.split(". ")
        if len(sentences) > 4:
            text = ". ".join(sentences[:4])

        return text.strip()

    # =========================
    # 🧠 VISUAL INTELLIGENCE
    # =========================
    def _build_visual_explanation(self, outfit, context):
        score_meta = outfit.get("score_meta", {}) if isinstance(outfit, dict) else {}
        if not isinstance(score_meta, dict):
            score_meta = {}
        reasons = score_meta.get("reasons", []) if isinstance(score_meta.get("reasons"), list) else []

        parts = []

        if "palette aligned" in reasons:
            parts.append("The colors work really well together.")

        if "clean aesthetic balance" in reasons:
            parts.append("The silhouette feels balanced and clean.")

        if "matches your style" in reasons:
            parts.append("This fits your personal style nicely.")

        if "aligned with your past choices" in reasons:
            parts.append("This aligns with what you usually like.")

        if not parts:
            parts.append("This comes together in a clean and easy way.")

        signals = context.get("signals", {}) if isinstance(context, dict) else {}
        if isinstance(signals, dict) and signals.get("weather_mode") == "hot":
            parts.append("This works well for warmer days.")

        return " ".join(parts).strip()

    def _bank_intelligence(self, outfit: dict, context: dict) -> str:
        """
        Surface the system's intelligence deterministically (no LLM required):
        - color harmony bank
        - print/pattern bank
        - silhouette bank
        - weather overlay bank
        - scorer reasons when present
        """
        try:
            outfit = outfit if isinstance(outfit, dict) else {}
            stable_key = str(outfit.get("combo_id") or outfit.get("id") or "outfit")

            items = outfit.get("refined_items") if isinstance(outfit.get("refined_items"), list) else outfit.get("items")
            if not isinstance(items, list):
                items = []

            patterns = [str((i or {}).get("pattern") or "").strip().lower() for i in items if isinstance(i, dict)]

            breakdown = outfit.get("score_breakdown") if isinstance(outfit.get("score_breakdown"), dict) else {}
            try:
                color_hint = float(breakdown.get("color_intelligence") or 0.0)
            except Exception:
                color_hint = 0.0
            try:
                silhouette_hint = float(breakdown.get("style_graph") or 0.0)
            except Exception:
                silhouette_hint = 0.0

            signals = context.get("signals", {}) or {}
            weather_mode = str(signals.get("weather_mode") or context.get("weather") or "").strip().lower()

            harmony_line = color_harmony_snippet(color_hint, key=stable_key)
            print_line = print_pattern_snippet(patterns, key=stable_key)
            silhouette_line = silhouette_snippet(silhouette_hint, key=stable_key)
            weather_line = weather_overlay_snippet(weather_mode, key=stable_key) if weather_mode else ""

            unified = outfit.get("unified_style") if isinstance(outfit.get("unified_style"), dict) else {}
            reasons = unified.get("reasons") if isinstance(unified.get("reasons"), list) else []
            reason_line = ""
            if reasons:
                reason_line = "Why it works: " + ", ".join([str(r).strip() for r in reasons if str(r).strip()][:2]) + "."

            return " ".join([x for x in [harmony_line, print_line, silhouette_line, weather_line, reason_line] if x]).strip()
        except Exception:
            return ""

    # =========================
    # 🧩 HELPERS
    # =========================
    def _safe_actions(self, context):
        try:
            return generate_followup_suggestions(context)
        except Exception:
            return ["Make it sharper", "Change colors", "Try another vibe"]

    def _llm_enabled(self):
        return os.getenv("ENABLE_LLM_SYNTHESIS", "true").lower() in ("1", "true")

    def _proactive_prefix(self, signals):
        if signals.get("suggestion_type") == "morning_outfit":
            return "Here’s something for your morning."
        if signals.get("suggestion_type") == "evening_outfit":
            return "This fits your evening."
        return ""

    def _reaction(self, outfit: dict | None = None):
        # Remove generic tone; let the intelligence-backed explanation carry the voice.
        return ""

    def _closer(self):
        return "Want a sharper or more relaxed version?"

    def _fallback(self, data: dict) -> str:
        if not data:
            return "I couldn’t find anything solid yet."

        chunks = []
        for k, v in data.items():
            if isinstance(v, dict):
                chunks.append(v.get("message") or v.get("summary") or "")
            else:
                chunks.append(str(v))

        return " ".join(chunks)


# singleton
response_assembler = ResponseAssembler()
