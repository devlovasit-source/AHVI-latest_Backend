class StyleEngine:
    """
    Minimal rules engine (safe stub)
    """

    def get_scoring_rules(self, style_dna, context):
        if not style_dna:
            return {
                "preferred_colors": [],
                "avoided_items": []
            }

        return {
            "preferred_colors": style_dna.get("preferred_colors", []),
            "avoided_items": style_dna.get("avoided_items", [])
        }


style_engine = StyleEngine()
