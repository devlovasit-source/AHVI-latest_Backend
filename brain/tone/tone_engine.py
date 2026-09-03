import logging
import os
import re
import json

_logger = logging.getLogger("ahvi.tone_engine")

_CONTEXT_MODE_ALIASES = {
    "chat": "conversation",
    "general": "conversation",
    "general_chat": "conversation",
    "style": "styling",
    "style_reasoning": "styling",
    "wardrobe": "styling",
    "daily_wear": "styling",
    "calendar": "planning",
    "planner": "planning",
}


def normalize_context_mode(value: str) -> str:
    """Map caller-specific labels to the tone engine's canonical contexts."""
    mode = str(value or "general").strip().lower().replace("-", "_")
    return _CONTEXT_MODE_ALIASES.get(mode, mode)


# Forbidden phrases that need a softened replacement rather than removal.
_FORBIDDEN_SOFTEN_MAP = {
    "here are some ideas": "A cleaner direction would be",
    "great choice!": "That works.",
    "great choice": "That works.",
    "this eats": "This feels strong.",
    "you ate that": "This feels confident.",
    "would you like me to": "I can",
    "not gonna lie": "Honestly,",
    "okay wait": "",
    "sure!": "",
    "absolutely!": "",
}


class ToneEngine:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "shared", "tone", "tone_engine.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.config = json.load(f).get("ahvi_tone_engine_v1", {})
        except Exception as e:
            print(f"WARN: Tone engine load failed: {e}")
            self.config = {}

    # =========================
    # MAIN ENTRY
    # =========================
    def apply(
        self,
        text: str,
        user_profile: dict = None,
        signals: dict = None,
        context: dict = None,
    ):

        if not text:
            return text

        user_profile = user_profile or {}
        signals = signals or {}
        context = context or {}

        user_memory = user_profile.get("memory", {})

        # -------------------------
        # 1. BASE DETECTION
        # -------------------------
        generation = self._detect_generation(user_profile)
        context_mode = normalize_context_mode(signals.get("context_mode", "general"))
        emotion = signals.get("emotion_state", "neutral")
        user_message_style = (
            signals.get("user_message_style", {})
            if isinstance(signals.get("user_message_style"), dict)
            else {}
        )

        context_rules = self.config.get("context_modes", {}).get(context_mode, {})
        emotion_rules = self.config.get("emotion_overrides", {}).get(emotion, {})
        generation_rules = self.config.get("generation_defaults", {}).get(
            generation, {}
        )
        limits = self._resolve_limits(
            generation_rules=generation_rules,
            context_rules=context_rules,
            emotion_rules=emotion_rules,
            user_message_style=user_message_style,
        )

        # -------------------------
        # 2. OUTFIT AESTHETIC
        # -------------------------
        aesthetic = context.get("aesthetic") or self._extract_outfit_aesthetic(context)

        # -------------------------
        # 3. LEARNED USER TONE
        # -------------------------
        learned_tone = user_memory.get("tone_preferences", {})

        # -------------------------
        # 4. APPLY BASE CONSTRAINTS
        # -------------------------
        text = self._apply_constraints(
            text,
            limits=limits,
            context_rules=context_rules,
            emotion_rules=emotion_rules,
        )

        # -------------------------
        # 5. APPLY OUTFIT TONE
        # -------------------------
        text = self._apply_outfit_tone(
            text,
            aesthetic,
            context_mode=context_mode,
            generation=generation,
            context_rules=context_rules,
            limits=limits,
        )

        # -------------------------
        # 6. APPLY LEARNED USER STYLE
        # -------------------------
        text = self._apply_user_preference(text, learned_tone, limits=limits)

        # -------------------------
        # 7. UPDATE MEMORY (FEEDBACK LOOP)
        # -------------------------
        updated_memory = self._update_learning(user_memory, signals, aesthetic)

        user_profile["memory"] = updated_memory

        return text

    def build_prompt_tone(self, user_profile: dict = None, signals: dict = None):
        """Compatibility shim for older services that ask for prompt-level tone."""
        user_profile = user_profile or {}
        signals = signals or {}
        generation = self._detect_generation(user_profile)
        context_mode = normalize_context_mode(signals.get("context_mode", "general"))
        rules = (
            self.config.get("context_modes", {}).get(context_mode, {})
            if isinstance(self.config, dict)
            else {}
        )
        return {
            "generation": generation,
            "context_mode": context_mode,
            "tone_instruction": rules.get("instruction")
            or "Warm, concise, practical AHVI styling tone.",
        }

    # =========================
    # ðŸ”¥ FEEDBACK LEARNING
    # =========================
    def _update_learning(self, memory, signals, aesthetic):

        memory = memory or {}
        prefs = memory.get(
            "tone_preferences", {"energy": "balanced", "style": "neutral"}
        )

        feedback = signals.get("feedback")
        engagement = signals.get("engagement_level")

        if not aesthetic:
            return memory

        # -------------------------
        # POSITIVE SIGNAL
        # -------------------------
        if feedback == "like" or engagement == "high":

            if aesthetic.get("energy") == "bold":
                prefs["energy"] = "bold"

            if aesthetic.get("vibe") == "minimal":
                prefs["style"] = "minimal"

            if aesthetic.get("vibe") == "street":
                prefs["style"] = "expressive"

        # -------------------------
        # NEGATIVE SIGNAL
        # -------------------------
        if feedback == "dislike":

            if aesthetic.get("energy") == "bold":
                prefs["energy"] = "soft"

        memory["tone_preferences"] = prefs
        return memory

    # =========================
    # ðŸ‘¤ USER STYLE APPLY
    # =========================
    def _apply_user_preference(self, text, prefs, limits: dict = None):
        limits = limits or {}

        if not prefs:
            return text

        # Targeted preference replacements only — avoid global word swaps that
        # produce awkward grammar (e.g. replacing every "good" → "strong").
        if prefs.get("energy") == "bold" and int(limits.get("humor", 0)) >= 15:
            text = text.replace("This is good.", "This feels strong.")
            text = text.replace("This looks good.", "This feels intentional.")
            text = text.replace("This is nice.", "This feels strong.")
        elif prefs.get("energy") == "soft":
            text = text.replace("This feels strong.", "This feels easy.")

        if prefs.get("style") == "minimal":
            text = text.replace("Try adding", "You could add")
        elif prefs.get("style") == "expressive" and int(limits.get("slang", 0)) >= 30:
            text = text.replace("clean finish", "clean finish with character")

        return text

    # =========================
    # ðŸŽ¨ OUTFIT AWARENESS
    # =========================
    def _extract_outfit_aesthetic(self, context):

        outfit = context.get("outfit_data", {}) or {}
        items = outfit.get("items", [])
        if not isinstance(items, list) or not items:
            return None

        colors = [str(i.get("color", "")).lower() for i in items]
        styles = [str(i.get("style", "")).lower() for i in items]

        dark = {"black", "navy", "charcoal"}
        light = {"white", "beige", "pastel"}

        dark_score = sum(1 for c in colors if c in dark)
        light_score = sum(1 for c in colors if c in light)

        return {
            "energy": "bold" if dark_score > light_score else "soft",
            "vibe": "street" if "street" in styles else "minimal",
            "structure": "sharp" if "formal" in styles else "relaxed",
        }

    def _apply_outfit_tone(
        self,
        text,
        aesthetic,
        context_mode: str = "general",
        generation: str = "other",
        context_rules: dict = None,
        limits: dict = None,
    ):
        context_rules = context_rules or {}
        limits = limits or {}

        if not aesthetic:
            return text

        if aesthetic.get("structure") == "sharp":
            text = text.replace("This works", "This is clean")

        allow_expressive = (
            context_mode in {"styling", "shopping"}
            and int(limits.get("slang", 0) or 0) >= 25
            and int(limits.get("sass", 0) or 0) >= 10
        )

        if (
            aesthetic.get("vibe") == "street"
            and allow_expressive
            and generation == "gen_z"
        ):
            text += " The silhouette carries a sharper street influence."

        if (
            aesthetic.get("vibe") == "minimal"
            and int(limits.get("slang", 0) or 0) <= 25
        ):
            text += " The finish stays clean and intentional."

        return text

    # =========================
    # BASE RULES
    # =========================
    def _apply_constraints(self, text, limits: dict, context_rules, emotion_rules):
        text = str(text or "")
        limits = limits or {}

        text = text.replace("!!", "!")
        text = text.replace("  ", " ").strip()
        text = self._remove_disallowed_slang(text)
        text = self._remove_forbidden_phrases(text)

        if emotion_rules.get("sentence_style") == "soft":
            text = text.replace("!", ".")

        if int(limits.get("slang", 0) or 0) <= 0:
            text = self._remove_slang(text)

        # Enforce emoji cap deterministically. Premium contexts (styling,
        # shopping, professional) ship with emoji_cap=0 and must not emit any.
        emoji_cap = int(limits.get("emoji", 0) or 0)
        text = self._enforce_emoji_cap(text, max_emojis=max(0, emoji_cap))

        max_exc = int(
            self.config.get("global_output_constraints", {})
            .get("grammar_and_punctuation", {})
            .get("max_exclamation_marks", 1)
            or 1
        )
        text = self._enforce_max_exclamations(text, max_exc=max(0, max_exc))
        return text

    def _remove_slang(self, text):
        slang_list = (
            self.config.get("slang_libraries", {})
            .get("gen_z", {})
            .get("approved_tokens", [])
        )
        for s in slang_list:
            text = text.replace(s, "")
        return text.strip()

    def _remove_forbidden_phrases(self, text: str) -> str:
        """Strip or soften forbidden generic-assistant/influencer phrases.

        Phrases live in the tone config under `forbidden_phrases`. Matching is
        case-insensitive; some phrases have explicit softer replacements
        (see _FORBIDDEN_SOFTEN_MAP), others are removed entirely.
        """
        if not text:
            return text
        forbidden = self.config.get("forbidden_phrases") or []
        if not isinstance(forbidden, list):
            return text
        out = text
        removed_any = False
        for raw in forbidden:
            phrase = str(raw or "").strip()
            if not phrase:
                continue
            soften = _FORBIDDEN_SOFTEN_MAP.get(phrase.lower(), "")
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            if pattern.search(out):
                out = pattern.sub(soften, out)
                removed_any = True
        if removed_any:
            try:
                _logger.info(
                    "ahvi.tone.forbidden_phrase_removed count=%d",
                    sum(1 for p in forbidden if str(p)),
                )
            except Exception:
                pass
        # Cleanup orphaned punctuation/whitespace left by removals.
        out = re.sub(r"\s{2,}", " ", out)
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        return out.strip()

    def _remove_disallowed_slang(self, text: str) -> str:
        disallowed = (
            self.config.get("slang_libraries", {})
            .get("gen_z", {})
            .get("disallowed_tokens", [])
        )
        out = text
        for token in disallowed:
            out = out.replace(token, "")
        return " ".join(out.split())

    _EMOJI_RE = re.compile(
        "["
        "\U0001F300-\U0001FAFF"  # symbols & pictographs, supplementals
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport
        "\U00002600-\U000027BF"  # misc symbols + dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # extended-A
        "]",
        flags=re.UNICODE,
    )

    def _enforce_emoji_cap(self, text: str, max_emojis: int = 0) -> str:
        if not text:
            return text
        if max_emojis <= 0:
            cleaned = self._EMOJI_RE.sub("", text)
            return re.sub(r"\s{2,}", " ", cleaned).strip()
        count = 0
        out_chars = []
        for ch in text:
            if self._EMOJI_RE.match(ch):
                if count < max_emojis:
                    out_chars.append(ch)
                    count += 1
                continue
            out_chars.append(ch)
        return re.sub(r"\s{2,}", " ", "".join(out_chars)).strip()

    def _enforce_max_exclamations(self, text: str, max_exc: int = 1) -> str:
        if max_exc < 0:
            return text
        count = 0
        out = []
        for ch in text:
            if ch == "!":
                if count < max_exc:
                    out.append(ch)
                count += 1
            else:
                out.append(ch)
        return "".join(out)

    def _resolve_limits(
        self,
        generation_rules: dict,
        context_rules: dict,
        emotion_rules: dict,
        user_message_style: dict,
    ) -> dict:
        generation_rules = generation_rules or {}
        context_rules = context_rules or {}
        emotion_rules = emotion_rules or {}
        user_message_style = user_message_style or {}

        slang = min(
            int(generation_rules.get("base_slang", 0) or 0),
            int(context_rules.get("slang_cap", 100) or 0),
        )
        humor = min(
            int(generation_rules.get("base_humor", 0) or 0),
            int(context_rules.get("humor_cap", 100) or 0),
        )
        sass = min(
            int(generation_rules.get("base_sass", 0) or 0),
            int(context_rules.get("sass_cap", 100) or 0),
        )
        emoji = min(
            int(generation_rules.get("base_emoji", 0) or 0),
            int(context_rules.get("emoji_cap", 100) or 0),
        )

        # Use explicit None checks so a `0` cap is not treated as "absent".
        if emotion_rules.get("slang_cap") is not None:
            slang = min(slang, int(emotion_rules.get("slang_cap") or 0))
        if emotion_rules.get("humor_cap") is not None:
            humor = min(humor, int(emotion_rules.get("humor_cap") or 0))
        if emotion_rules.get("sass_cap") is not None:
            sass = min(sass, int(emotion_rules.get("sass_cap") or 0))
        if emotion_rules.get("emoji_cap") is not None:
            emoji = min(emoji, int(emotion_rules.get("emoji_cap") or 0))

        slang += int(emotion_rules.get("slang_boost", 0) or 0)
        humor += int(emotion_rules.get("humor_boost", 0) or 0)
        sass += int(emotion_rules.get("sass_boost", 0) or 0)
        emoji += int(emotion_rules.get("emoji_boost", 0) or 0)

        mirror_slang_map = self.config.get("mirroring_rules", {}).get(
            "slang_presence", {}
        )
        slang_bucket = str(user_message_style.get("slang_presence") or "").lower()
        if slang_bucket in mirror_slang_map:
            mirror_max = int(
                (mirror_slang_map.get(slang_bucket) or {}).get(
                    "assistant_slang_tokens_max", 0
                )
                or 0
            )
            slang = min(slang, mirror_max * 25)

        mirror_emoji_map = self.config.get("mirroring_rules", {}).get(
            "emoji_density", {}
        )
        emoji_bucket = str(user_message_style.get("emoji_density") or "").lower()
        if emoji_bucket in mirror_emoji_map:
            mirror_emoji = int(
                (mirror_emoji_map.get(emoji_bucket) or {}).get(
                    "assistant_max_emojis", 0
                )
                or 0
            )
            emoji = min(emoji, mirror_emoji)

        return {
            "slang": max(0, min(slang, 100)),
            "humor": max(0, min(humor, 100)),
            "sass": max(0, min(sass, 100)),
            "emoji": max(0, min(emoji, 4)),
        }

    # =========================
    # GENERATION
    # =========================
    def _detect_generation(self, user_profile):
        if not user_profile:
            return "other"

        # Prefer explicit age when provided by client profile payloads.
        try:
            age = int(user_profile.get("age"))
            current_year = 2026
            year = current_year - age
        except Exception:
            if not user_profile.get("dob_iso"):
                return "other"
            try:
                year = int(str(user_profile["dob_iso"]).split("-")[0])
            except Exception:
                return "other"

        buckets = self.config.get("generation_buckets", {})

        for name, r in buckets.items():
            if r["dob_year_min"] <= year <= r["dob_year_max"]:
                return name

        return "other"


# Singleton
tone_engine = ToneEngine()
