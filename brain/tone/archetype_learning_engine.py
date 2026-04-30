class ArchetypeLearningEngine:

    def update(self, user_memory: dict, signals: dict) -> dict:
        """
        Adaptive learning engine for archetype + interaction style
        """

        memory = user_memory or {}

        # -------------------------
        # 🔥 INIT DEFAULTS
        # -------------------------
        scores = memory.get("archetype_scores", {
            "stylist": 0.34,
            "best_friend": 0.33,
            "advisor": 0.33
        })

        interaction_style = memory.get("interaction_style", {
            "likes_slang": False,
            "likes_short": True,
            "likes_hype": False
        })

        feedback = signals.get("feedback")              # like / dislike
        engagement = signals.get("engagement_level")    # low / medium / high
        last_archetype = signals.get("last_archetype")

        message_length = signals.get("response_length")  # short / long
        slang_used = signals.get("slang_used", False)

        if not last_archetype:
            return memory

        # -------------------------
        # 🔥 SCORE UPDATE
        # -------------------------
        delta = 0.0

        if feedback == "like":
            delta += 0.15
        elif feedback == "dislike":
            delta -= 0.2

        if engagement == "high":
            delta += 0.1
        elif engagement == "low":
            delta -= 0.1

        scores[last_archetype] = max(0.01, scores.get(last_archetype, 0) + delta)

        # -------------------------
        # 🔥 DECAY (important)
        # -------------------------
        for k in scores:
            if k != last_archetype:
                scores[k] *= 0.98  # slight decay

        # -------------------------
        # 🔥 NORMALIZE
        # -------------------------
        total = sum(scores.values())
        if total > 0:
            for k in scores:
                scores[k] = round(scores[k] / total, 3)

        # -------------------------
        # 🔥 UPDATE INTERACTION STYLE
        # -------------------------
        if slang_used:
            interaction_style["likes_slang"] = True

        if message_length == "short" and engagement == "high":
            interaction_style["likes_short"] = True

        if feedback == "like" and last_archetype == "best_friend":
            interaction_style["likes_hype"] = True

        # -------------------------
        # 🔥 PREFERRED ARCHETYPE
        # -------------------------
        preferred = max(scores, key=scores.get)

        # -------------------------
        # 🔥 SAVE BACK
        # -------------------------
        memory["archetype_scores"] = scores
        memory["preferred_archetype"] = preferred
        memory["interaction_style"] = interaction_style

        return memory


# Singleton
archetype_learning_engine = ArchetypeLearningEngine()
