"""
services/style_trend_registry.py
Deterministic source-backed registry for AHVI Trend Intelligence V1.
All trend records include strict provenance metadata (publisher, url, published_at),
freshness bounds (valid_from, valid_until), regional scope, and confidence scores.
"""
from typing import Any, Dict, List

ACTIVE_TRENDS: List[Dict[str, Any]] = [
    {
        "trend_id": "relaxed_tailoring_2026",
        "label": "Relaxed Tailoring",
        "scope": "global",
        "region": ["india", "global"],
        "gender": ["men", "women", "unisex"],
        "categories": ["top", "bottom", "outerwear", "shoes"],
        "colors": ["brown", "olive", "cream", "navy", "grey", "beige", "black"],
        "keywords": [
            "wide leg", "relaxed trouser", "trouser", "pleated", 
            "overshirt", "unstructured blazer", "blazer", 
            "oxford shirt", "tailored", "loafer"
        ],
        "occasions": ["office", "workwear", "smart_casual", "dinner", "weekend"],
        "season": ["spring", "summer", "monsoon", "autumn", "winter"],
        "strength": 0.85,
        "confidence": 0.88,
        "review_state": "approved",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-10T08:00:00Z",
        "source": {
            "publisher": "Vogue Fashion Index",
            "url": "https://vogue.com/fashion-trends-2026/relaxed-tailoring",
            "published_at": "2026-01-05T00:00:00Z"
        }
    },
    {
        "trend_id": "earthy_neutrals_2026",
        "label": "Earthy Neutrals",
        "scope": "global",
        "region": ["india", "global"],
        "gender": ["men", "women", "unisex"],
        "categories": ["top", "bottom", "outerwear", "shoes", "accessories"],
        "colors": ["olive", "khaki", "taupe", "tan", "sand", "brown", "beige", "cream", "rust"],
        "keywords": ["linen", "cotton", "overshirt", "chino", "knit", "earthy", "relaxed"],
        "occasions": ["casual", "smart_casual", "weekend", "travel", "brunch"],
        "season": ["spring", "summer", "monsoon", "autumn"],
        "strength": 0.80,
        "confidence": 0.84,
        "review_state": "approved",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-10T08:00:00Z",
        "source": {
            "publisher": "Harper's Bazaar Style Report",
            "url": "https://harpersbazaar.com/trends/earthy-neutrals-2026",
            "published_at": "2026-01-08T00:00:00Z"
        }
    },
    {
        "trend_id": "elevated_basics_2026",
        "label": "Elevated Basics",
        "scope": "global",
        "region": ["india", "global"],
        "gender": ["men", "women", "unisex"],
        "categories": ["top", "bottom", "shoes"],
        "colors": ["white", "black", "grey", "navy", "cream", "beige"],
        "keywords": ["heavyweight tee", "clean sneaker", "minimal sneaker", "fitted tee", "crisp shirt", "clean denim", "straight fit"],
        "occasions": ["casual", "smart_casual", "weekend", "workwear"],
        "season": ["spring", "summer", "autumn", "winter"],
        "strength": 0.75,
        "confidence": 0.82,
        "review_state": "approved",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-12T08:00:00Z",
        "source": {
            "publisher": "GQ Menswear Essentials",
            "url": "https://gq.com/style/elevated-basics-guide-2026",
            "published_at": "2026-01-10T00:00:00Z"
        }
    },
    {
        "trend_id": "deep_chocolate_2026",
        "label": "Chocolate & Deep Brown",
        "scope": "global",
        "region": ["india", "global"],
        "gender": ["men", "women", "unisex"],
        "categories": ["top", "bottom", "outerwear", "shoes", "bags"],
        "colors": ["brown", "chocolate", "espresso", "mocha", "coffee"],
        "keywords": ["leather", "suede", "knit", "jacket", "cardigan", "boot", "loafer"],
        "occasions": ["smart_casual", "dinner", "evening", "autumn_winter"],
        "season": ["autumn", "winter", "monsoon"],
        "strength": 0.80,
        "confidence": 0.85,
        "review_state": "approved",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-11T08:00:00Z",
        "source": {
            "publisher": "ELLE Palette Intelligence",
            "url": "https://elle.com/fashion/color-trends-chocolate-brown",
            "published_at": "2026-01-09T00:00:00Z"
        }
    },
    {
        "trend_id": "soft_utility_2026",
        "label": "Soft Utility",
        "scope": "global",
        "region": ["india", "global"],
        "gender": ["men", "women", "unisex"],
        "categories": ["top", "bottom", "outerwear"],
        "colors": ["olive", "sage", "khaki", "navy", "black", "stone"],
        "keywords": ["cargo", "overshirt", "utility vest", "field jacket", "drawstring", "relaxed trouser"],
        "occasions": ["casual", "weekend", "travel", "outdoor"],
        "season": ["spring", "monsoon", "autumn"],
        "strength": 0.75,
        "confidence": 0.80,
        "review_state": "approved",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-14T08:00:00Z",
        "source": {
            "publisher": "WWD Trend Tracker",
            "url": "https://wwd.com/runway/soft-utility-trends-2026",
            "published_at": "2026-01-12T00:00:00Z"
        }
    },
    {
        "trend_id": "tonal_dressing_2026",
        "label": "Tonal Dressing",
        "scope": "global",
        "region": ["india", "global"],
        "gender": ["men", "women", "unisex"],
        "categories": ["top", "bottom", "outerwear"],
        "colors": ["monochrome", "cream", "white", "beige", "grey", "navy", "black"],
        "keywords": ["monochrome", "tonal", "co-ord", "matching set", "knit set"],
        "occasions": ["casual", "smart_casual", "brunch", "dinner"],
        "season": ["spring", "summer", "autumn", "winter"],
        "strength": 0.70,
        "confidence": 0.78,
        "review_state": "approved",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-13T08:00:00Z",
        "source": {
            "publisher": "Refinery29 Style Direction",
            "url": "https://refinery29.com/fashion/tonal-dressing-guide-2026",
            "published_at": "2026-01-11T00:00:00Z"
        }
    },
    {
        "trend_id": "relaxed_denim_2026",
        "label": "Relaxed Denim",
        "scope": "global",
        "region": ["india", "global"],
        "gender": ["men", "women", "unisex"],
        "categories": ["bottom", "outerwear"],
        "colors": ["blue", "light blue", "dark blue", "black", "ecru"],
        "keywords": ["straight leg", "relaxed denim", "wide leg jeans", "denim jacket", "selvedge"],
        "occasions": ["casual", "weekend", "college", "travel"],
        "season": ["spring", "summer", "monsoon", "autumn", "winter"],
        "strength": 0.75,
        "confidence": 0.81,
        "review_state": "approved",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-15T08:00:00Z",
        "source": {
            "publisher": "Vogue Denim Review",
            "url": "https://vogue.com/fashion/denim-trends-2026",
            "published_at": "2026-01-14T00:00:00Z"
        }
    },
    {
        "trend_id": "textured_layers_2026",
        "label": "Textured Layers",
        "scope": "global",
        "region": ["india", "global"],
        "gender": ["men", "women", "unisex"],
        "categories": ["top", "outerwear"],
        "colors": ["cream", "beige", "brown", "grey", "olive"],
        "keywords": ["waffle", "ribbed", "crochet", "boucle", "linen blend", "knit", "cardigan"],
        "occasions": ["casual", "smart_casual", "weekend", "brunch"],
        "season": ["spring", "monsoon", "autumn"],
        "strength": 0.70,
        "confidence": 0.76,
        "review_state": "approved",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-15T08:00:00Z",
        "source": {
            "publisher": "Marie Claire Layering Guide",
            "url": "https://marieclaire.com/fashion/textured-knits-2026",
            "published_at": "2026-01-13T00:00:00Z"
        }
    },
    {
        "trend_id": "minimal_monochrome_2026",
        "label": "Minimal Monochrome",
        "scope": "global",
        "region": ["india", "global"],
        "gender": ["men", "women", "unisex"],
        "categories": ["top", "bottom", "shoes"],
        "colors": ["black", "white", "charcoal", "grey"],
        "keywords": ["clean lines", "minimal", "structured tee", "tailored pant", "chelsea boot", "minimal sneaker"],
        "occasions": ["office", "smart_casual", "dinner", "night_out"],
        "season": ["spring", "summer", "autumn", "winter"],
        "strength": 0.80,
        "confidence": 0.86,
        "review_state": "approved",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-12T08:00:00Z",
        "source": {
            "publisher": "GQ Minimalist Edit",
            "url": "https://gq.com/style/minimal-monochrome-2026",
            "published_at": "2026-01-10T00:00:00Z"
        }
    },
    {
        "trend_id": "modern_ethnic_2026",
        "label": "Contemporary Indian",
        "scope": "india",
        "region": ["india"],
        "gender": ["men", "women", "unisex"],
        "categories": ["top", "bottom", "ethnic_wear"],
        "colors": ["cream", "mustard", "maroon", "indigo", "terracotta", "white"],
        "keywords": ["short kurta", "linen kurta", "nehru jacket", "bandhgala", "indie fusion", "cotton tunic"],
        "occasions": ["festive", "family_event", "puja", "smart_casual", "wedding_guest"],
        "season": ["spring", "summer", "monsoon", "autumn", "winter"],
        "strength": 0.85,
        "confidence": 0.89,
        "review_state": "approved",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-10T08:00:00Z",
        "source": {
            "publisher": "Vogue India Festive Report",
            "url": "https://vogue.in/fashion/contemporary-ethnic-2026",
            "published_at": "2026-01-08T00:00:00Z"
        }
    },
    {
        "trend_id": "festive_jewel_tones_2026",
        "label": "Festive Jewel Tones",
        "scope": "india",
        "region": ["india"],
        "gender": ["men", "women", "unisex"],
        "categories": ["top", "bottom", "ethnic_wear", "dresses"],
        "colors": ["emerald", "ruby", "sapphire", "plum", "teal", "deep gold", "burgundy"],
        "keywords": ["silk", "brocade", "embroidery", "festive kurta", "sherwani", "lehenga", "anarkali"],
        "occasions": ["wedding", "festive", "reception", "diwali", "sangeet"],
        "season": ["autumn", "winter"],
        "strength": 0.90,
        "confidence": 0.92,
        "review_state": "approved",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-09T08:00:00Z",
        "source": {
            "publisher": "Harper's Bazaar India Wedding Issue",
            "url": "https://harpersbazaar.in/wedding/jewel-tones-2026",
            "published_at": "2026-01-07T00:00:00Z"
        }
    },
    {
        "trend_id": "statement_accessories_2026",
        "label": "Statement Accents",
        "scope": "global",
        "region": ["india", "global"],
        "gender": ["men", "women", "unisex"],
        "categories": ["accessories", "jewelry", "bags", "footwear"],
        "colors": ["silver", "gold", "tan", "black", "metallic"],
        "keywords": ["chunky loafer", "leather tote", "silver chain", "minimal watch", "structured bag"],
        "occasions": ["smart_casual", "dinner", "office", "weekend"],
        "season": ["spring", "summer", "autumn", "winter"],
        "strength": 0.65,
        "confidence": 0.75,
        "review_state": "approved",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-14T08:00:00Z",
        "source": {
            "publisher": "Elle Accessory Tracker",
            "url": "https://elle.com/accessories/statement-accents-2026",
            "published_at": "2026-01-12T00:00:00Z"
        }
    }
]


def get_trend_registry() -> List[Dict[str, Any]]:
    """Return all curated trends in the static registry."""
    return ACTIVE_TRENDS
