import os
import json


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
    def apply(self, text: str, user_profile: dict = None, signals: dict = None, context: dict = None):

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
        context_mode = signals.get("context_mode", "general")
        emotion = signals.get("emotion_state", "neutral")
        user_message_style = signals.get("user_message_style", {}) if isinstance(signals.get("user_message_style"), dict) else {}

        context_rules = self.config.get("context_modes", {}).get(context_mode, {})
        emotion_rules = self.config.get("emotion_overrides", {}).get(emotion, {})
        generation_rules = self.config.get("generation_defaults", {}).get(generation, {})
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
        text = self._apply_constraints(text, limits=limits, context_rules=context_rules, emotion_rules=emotion_rules)

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

    # =========================
    # 🔥 FEEDBACK LEARNING
    # =========================
    def _update_learning(self, memory, signals, aesthetic):

        memory = memory or {}
        prefs = memory.get("tone_preferences", {
            "energy": "balanced",
            "style": "neutral"
        })

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
    # 👤 USER STYLE APPLY
    # =========================
    def _apply_user_preference(self, text, prefs, limits: dict = None):
        limits = limits or {}

        if not prefs:
            return text

        # Keep preference application deterministic and low-noise.
        if prefs.get("energy") == "bold" and int(limits.get("humor", 0)) >= 15:
            text = text.replace("nice", "strong")
            text = text.replace("good", "strong")
        elif prefs.get("energy") == "soft":
            text = text.replace("strong", "easy")

        if prefs.get("style") == "minimal":
            text = text.replace("Try adding", "You could add")
        elif prefs.get("style") == "expressive" and int(limits.get("slang", 0)) >= 30:
            text = text.replace("clean", "clean with character")

        return text

    # =========================
    # 🎨 OUTFIT AWARENESS
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
            "structure": "sharp" if "formal" in styles else "relaxed"
        }

    def _apply_outfit_tone(self, text, aesthetic, context_mode: str = "general", generation: str = "other", context_rules: dict = None, limits: dict = None):
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

        if aesthetic.get("vibe") == "street" and allow_expressive and generation == "gen_z":
            text += " This lands with confident street energy."

        if aesthetic.get("vibe") == "minimal" and int(limits.get("slang", 0) or 0) <= 25:
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

        if emotion_rules.get("sentence_style") == "soft":
            text = text.replace("!", ".")

        if int(limits.get("slang", 0) or 0) <= 0:
            text = self._remove_slang(text)

        max_exc = int(self.config.get("global_output_constraints", {}).get("grammar_and_punctuation", {}).get("max_exclamation_marks", 1) or 1)
        text = self._enforce_max_exclamations(text, max_exc=max(0, max_exc))
        return text

    def _remove_slang(self, text):
        slang_list = self.config.get("slang_libraries", {}).get("gen_z", {}).get("approved_tokens", [])
        for s in slang_list:
            text = text.replace(s, "")
        return text.strip()

    def _remove_disallowed_slang(self, text: str) -> str:
        disallowed = self.config.get("slang_libraries", {}).get("gen_z", {}).get("disallowed_tokens", [])
        out = text
        for token in disallowed:
            out = out.replace(token, "")
        return " ".join(out.split())

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

    def _resolve_limits(self, generation_rules: dict, context_rules: dict, emotion_rules: dict, user_message_style: dict) -> dict:
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

        if "slang_cap" in emotion_rules:
            slang = min(slang, int(emotion_rules.get("slang_cap", slang) or slang))
        if "humor_cap" in emotion_rules:
            humor = min(humor, int(emotion_rules.get("humor_cap", humor) or humor))
        if "sass_cap" in emotion_rules:
            sass = min(sass, int(emotion_rules.get("sass_cap", sass) or sass))
        if "emoji_cap" in emotion_rules:
            emoji = min(emoji, int(emotion_rules.get("emoji_cap", emoji) or emoji))

        slang += int(emotion_rules.get("slang_boost", 0) or 0)
        humor += int(emotion_rules.get("humor_boost", 0) or 0)
        sass += int(emotion_rules.get("sass_boost", 0) or 0)
        emoji += int(emotion_rules.get("emoji_boost", 0) or 0)

        mirror_slang_map = self.config.get("mirroring_rules", {}).get("slang_presence", {})
        slang_bucket = str(user_message_style.get("slang_presence") or "").lower()
        if slang_bucket in mirror_slang_map:
            mirror_max = int((mirror_slang_map.get(slang_bucket) or {}).get("assistant_slang_tokens_max", 0) or 0)
            slang = min(slang, mirror_max * 25)

        mirror_emoji_map = self.config.get("mirroring_rules", {}).get("emoji_density", {})
        emoji_bucket = str(user_message_style.get("emoji_density") or "").lower()
        if emoji_bucket in mirror_emoji_map:
            mirror_emoji = int((mirror_emoji_map.get(emoji_bucket) or {}).get("assistant_max_emojis", 0) or 0)
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
