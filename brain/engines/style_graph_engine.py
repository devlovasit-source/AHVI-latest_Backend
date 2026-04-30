from itertools import combinations
from typing import Any, Dict, List


class StyleGraphEngine:
    """
    🔥 ELITE GRAPH ENGINE

    - color harmony + contrast
    - fabric logic
    - silhouette compatibility
    - outfit intelligence scoring
    """

    def build_graph(self, wardrobe: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:

        items: List[Dict[str, Any]] = []

        for key in ("tops", "bottoms", "shoes"):
            for item in wardrobe.get(key, []):
                if not isinstance(item, dict):
                    continue

                item_id = str(item.get("id") or item.get("name") or "")
                if not item_id:
                    continue

                items.append({
                    "id": item_id,
                    "type": str(item.get("type", "")).lower(),
                    "color": str(item.get("color", "")).lower(),
                    "fabric": str(item.get("fabric", "")).lower(),
                    "fit": str(item.get("fit", "")).lower(),  # NEW
                })

        edges = []
        edge_map = {}

        for left, right in combinations(items, 2):
            weight = self._edge_weight(left, right)
            if weight <= 0:
                continue

            key = self._pair_key(left["id"], right["id"])
            edge_map[key] = weight

            edges.append({
                "from": left["id"],
                "to": right["id"],
                "weight": weight
            })

        return {
            "nodes": items,
            "edges": edges,
            "edge_map": edge_map
        }

    # =========================
    # PUBLIC
    # =========================
    def pair_weight(self, graph: Dict[str, Any], a: str, b: str) -> float:
        edge_map = graph.get("edge_map", {})
        return float(edge_map.get(self._pair_key(a, b), 0.0))

    # =========================
    # INTERNAL
    # =========================
    def _pair_key(self, a: str, b: str) -> str:
        x, y = sorted([str(a), str(b)])
        return f"{x}|{y}"

    def _edge_weight(self, l: Dict[str, str], r: Dict[str, str]) -> float:
        score = 0.0

        # 🎨 COLOR LOGIC
        if l["color"] and r["color"]:
            if l["color"] == r["color"]:
                score += 0.6  # harmony
            elif self._is_neutral(l["color"]) or self._is_neutral(r["color"]):
                score += 0.4  # neutral pairing
            else:
                score += 0.3  # contrast

        # 🧵 FABRIC LOGIC
        if l["fabric"] and r["fabric"]:
            if l["fabric"] == r["fabric"]:
                score += 0.4
            elif self._is_good_mix(l["fabric"], r["fabric"]):
                score += 0.3

        # 👕 TYPE COMPATIBILITY
        if self._is_complementary(l["type"], r["type"]):
            score += 1.2

        # 🧍 SILHOUETTE LOGIC
        if self._silhouette_balance(l.get("fit"), r.get("fit")):
            score += 0.6

        return score

    # =========================
    # HELPERS
    # =========================
    def _is_neutral(self, color: str) -> bool:
        return color in ["black", "white", "grey", "beige", "navy"]

    def _is_good_mix(self, f1: str, f2: str) -> bool:
        pairs = [
            ("denim", "cotton"),
            ("wool", "cotton"),
            ("linen", "cotton"),
        ]
        return (f1, f2) in pairs or (f2, f1) in pairs

    def _is_complementary(self, t1: str, t2: str) -> bool:
        pairs = [
            ("shirt", "trousers"),
            ("tshirt", "jeans"),
            ("top", "bottom"),
            ("kurta", "churidar"),
        ]
        return any((a in t1 and b in t2) or (a in t2 and b in t1) for a, b in pairs)

    def _silhouette_balance(self, f1: str, f2: str) -> bool:
        if not f1 or not f2:
            return False

        # fitted + relaxed = good
        combos = [
            ("slim", "relaxed"),
            ("oversized", "slim"),
        ]

        return any((a in f1 and b in f2) or (a in f2 and b in f1) for a, b in combos)


# Singleton
style_graph_engine = StyleGraphEngine()
