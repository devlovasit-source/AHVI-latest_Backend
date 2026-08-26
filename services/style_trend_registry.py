"""
services/style_trend_registry.py
Deterministic source-backed registry for AHVI Trend Intelligence V1.

Schema, provenance concept, and freshness-window fields are adapted from
PR #48 (feat/trend-intelligence-dynamic, f0a57eb). The DATA is not: every
record below was checked against its claimed `source.url` on 2026-08-26 and
none passed independent verification (see `verification_status` per
record). This registry therefore ships with zero VERIFIED records for V1 --
the mechanism is proven end-to-end, but no trend is allowed to present
itself as source-backed until a human editor actually confirms a source.

Verification method used: HTTP GET against each claimed url.
  - 10/12 returned 404 (dead link) -> verification_status="DEAD_LINK"
  - 2/12 returned 200 but failed content verification -> "UNCONFIRMED_CONTENT"
      * relaxed_tailoring_2026: the live page title is "The Key Summer 2026
        Trends..." -- a different season/topic than what this record claims.
      * earthy_neutrals_2026: the claimed url used a placeholder numeric ID
        (.../a60000000/...) that Harper's Bazaar's CMS silently redirected
        to an unrelated real article ID -- not evidence the redirect target
        actually discusses "Earthy Neutrals".
  - Every record in PR #48 shared the exact same verified_at timestamp
    (2026-02-01T10:00:00Z) across 12 different publishers -- itself strong
    evidence verified_at was never a real, independent verification event.
    verified_at is therefore set to None here rather than kept as invented
    metadata (do not resurrect the PR #48 timestamp).

`review_state` is downgraded from PR48's "approved" to "unverified" for
every record for the same reason: no human/process has actually approved
these sources. TrendContextService.is_trend_valid() hard-gates on both
review_state=="approved" AND verification_status=="VERIFIED", so nothing
here is reachable at runtime -- get_active_trends() returns [] until a real
record is added with genuine, checked provenance.
"""
from typing import Any, Dict, List

ACTIVE_TRENDS: List[Dict[str, Any]] = [
    {
        "trend_id": "relaxed_tailoring_2026",
        "label": "Relaxed Tailoring",
        "scope": "global",
        "region": ["global"],
        "gender": ["male", "female", "unisex"],
        "categories": ["top", "bottom", "outerwear", "shoes"],
        "colors": ["brown", "olive", "cream", "navy", "grey", "beige", "black"],
        "keywords": [
            "wide leg", "relaxed trouser", "trouser", "pleated",
            "overshirt", "unstructured blazer", "blazer",
            "oxford shirt", "tailored", "loafer",
        ],
        "occasions": ["office", "workwear", "smart_casual", "dinner", "weekend"],
        "season": ["spring", "summer", "monsoon", "autumn", "winter"],
        "strength": 0.85,
        "confidence": 0.88,
        "review_state": "unverified",
        "verification_status": "UNCONFIRMED_CONTENT",
        "verification_note": "URL resolves (HTTP 200) but live title is 'The Key Summer 2026 Trends...' -- topic/season mismatch with this record's claims.",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-10T08:00:00Z",
        "verified_at": None,
        "source": {
            "publisher": "Vogue Fashion Index",
            "url": "https://www.vogue.com/article/spring-2026-fashion-trends",
            "published_at": "2026-01-15T00:00:00Z",
        },
    },
    {
        "trend_id": "earthy_neutrals_2026",
        "label": "Earthy Neutrals",
        "scope": "global",
        "region": ["global"],
        "gender": ["male", "female", "unisex"],
        "categories": ["top", "bottom", "outerwear", "shoes", "accessories"],
        "colors": ["olive", "khaki", "taupe", "tan", "sand", "brown", "beige", "cream", "rust"],
        "keywords": ["linen", "cotton", "overshirt", "chino", "knit", "earthy", "relaxed"],
        "occasions": ["casual", "smart_casual", "weekend", "travel", "brunch"],
        "season": ["spring", "summer", "monsoon", "autumn"],
        "strength": 0.80,
        "confidence": 0.84,
        "review_state": "unverified",
        "verification_status": "UNCONFIRMED_CONTENT",
        "verification_note": "Claimed URL used a placeholder numeric ID (a60000000) that the publisher's CMS silently redirected elsewhere -- redirect target's actual content was not confirmed.",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-10T08:00:00Z",
        "verified_at": None,
        "source": {
            "publisher": "Harper's Bazaar Style Report",
            "url": "https://www.harpersbazaar.com/fashion/trends/a60000000/spring-2026-fashion-trends/",
            "published_at": "2026-01-20T00:00:00Z",
        },
    },
    {
        "trend_id": "elevated_basics_2026",
        "label": "Elevated Basics",
        "scope": "global",
        "region": ["global"],
        "gender": ["male", "female", "unisex"],
        "categories": ["top", "bottom", "shoes"],
        "colors": ["white", "black", "grey", "navy", "cream", "beige"],
        "keywords": ["heavyweight tee", "clean sneaker", "minimal sneaker", "fitted tee", "crisp shirt", "clean denim", "straight fit"],
        "occasions": ["casual", "smart_casual", "weekend", "workwear"],
        "season": ["spring", "summer", "autumn", "winter"],
        "strength": 0.75,
        "confidence": 0.82,
        "review_state": "unverified",
        "verification_status": "DEAD_LINK",
        "verification_note": "HTTP 404 on the claimed source URL (checked 2026-08-26).",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-12T08:00:00Z",
        "verified_at": None,
        "source": {
            "publisher": "GQ Menswear Essentials",
            "url": "https://www.gq.com/story/mens-fashion-trends-2026",
            "published_at": "2026-01-10T00:00:00Z",
        },
    },
    {
        "trend_id": "deep_chocolate_2026",
        "label": "Chocolate & Deep Brown",
        "scope": "global",
        "region": ["global"],
        "gender": ["male", "female", "unisex"],
        "categories": ["top", "bottom", "outerwear", "shoes", "bags"],
        "colors": ["brown", "chocolate", "espresso", "mocha", "coffee"],
        "keywords": ["leather", "suede", "knit", "jacket", "cardigan", "boot", "loafer"],
        "occasions": ["smart_casual", "dinner", "evening", "autumn_winter"],
        "season": ["autumn", "winter", "monsoon"],
        "strength": 0.80,
        "confidence": 0.85,
        "review_state": "unverified",
        "verification_status": "DEAD_LINK",
        "verification_note": "HTTP 404 on the claimed source URL (checked 2026-08-26).",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-11T08:00:00Z",
        "verified_at": None,
        "source": {
            "publisher": "ELLE Palette Intelligence",
            "url": "https://www.elle.com/fashion/trend-reports/a60000000/fashion-trends-2026/",
            "published_at": "2026-01-12T00:00:00Z",
        },
    },
    {
        "trend_id": "soft_utility_2026",
        "label": "Soft Utility",
        "scope": "global",
        "region": ["global"],
        "gender": ["male", "female", "unisex"],
        "categories": ["top", "bottom", "outerwear"],
        "colors": ["olive", "sage", "khaki", "navy", "black", "stone"],
        "keywords": ["cargo", "overshirt", "utility vest", "field jacket", "drawstring", "relaxed trouser"],
        "occasions": ["casual", "weekend", "travel", "outdoor"],
        "season": ["spring", "monsoon", "autumn"],
        "strength": 0.75,
        "confidence": 0.80,
        "review_state": "unverified",
        "verification_status": "DEAD_LINK",
        "verification_note": "HTTP 404 on the claimed source URL (checked 2026-08-26).",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-14T08:00:00Z",
        "verified_at": None,
        "source": {
            "publisher": "WWD Trend Tracker",
            "url": "https://wwd.com/fashion-news/fashion-scoops/utility-fashion-trends-2026",
            "published_at": "2026-01-14T00:00:00Z",
        },
    },
    {
        "trend_id": "tonal_dressing_2026",
        "label": "Tonal Dressing",
        "scope": "global",
        "region": ["global"],
        "gender": ["male", "female", "unisex"],
        "categories": ["top", "bottom", "outerwear"],
        "colors": ["monochrome", "cream", "white", "beige", "grey", "navy", "black"],
        "keywords": ["monochrome", "tonal", "co-ord", "matching set", "knit set"],
        "occasions": ["casual", "smart_casual", "brunch", "dinner"],
        "season": ["spring", "summer", "autumn", "winter"],
        "strength": 0.70,
        "confidence": 0.78,
        "review_state": "unverified",
        "verification_status": "DEAD_LINK",
        "verification_note": "HTTP 404 on the claimed source URL (checked 2026-08-26).",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-13T08:00:00Z",
        "verified_at": None,
        "source": {
            "publisher": "Refinery29 Style Direction",
            "url": "https://www.refinery29.com/en-us/monochrome-tonal-trends-2026",
            "published_at": "2026-01-11T00:00:00Z",
        },
    },
    {
        "trend_id": "relaxed_denim_2026",
        "label": "Relaxed Denim",
        "scope": "global",
        "region": ["global"],
        "gender": ["male", "female", "unisex"],
        "categories": ["bottom", "outerwear"],
        "colors": ["blue", "light blue", "dark blue", "black", "ecru"],
        "keywords": ["straight leg", "relaxed denim", "wide leg jeans", "denim jacket", "selvedge"],
        "occasions": ["casual", "weekend", "college", "travel"],
        "season": ["spring", "summer", "monsoon", "autumn", "winter"],
        "strength": 0.75,
        "confidence": 0.81,
        "review_state": "unverified",
        "verification_status": "DEAD_LINK",
        "verification_note": "HTTP 404 on the claimed source URL (checked 2026-08-26).",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-15T08:00:00Z",
        "verified_at": None,
        "source": {
            "publisher": "Vogue Denim Review",
            "url": "https://www.vogue.com/article/best-denim-jeans-trends-2026",
            "published_at": "2026-01-14T00:00:00Z",
        },
    },
    {
        "trend_id": "textured_layers_2026",
        "label": "Textured Layers",
        "scope": "global",
        "region": ["global"],
        "gender": ["male", "female", "unisex"],
        "categories": ["top", "outerwear"],
        "colors": ["cream", "beige", "brown", "grey", "olive"],
        "keywords": ["waffle", "ribbed", "crochet", "boucle", "linen blend", "knit", "cardigan"],
        "occasions": ["casual", "smart_casual", "weekend", "brunch"],
        "season": ["spring", "monsoon", "autumn"],
        "strength": 0.70,
        "confidence": 0.76,
        "review_state": "unverified",
        "verification_status": "DEAD_LINK",
        "verification_note": "HTTP 404 on the claimed source URL (checked 2026-08-26).",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-15T08:00:00Z",
        "verified_at": None,
        "source": {
            "publisher": "Marie Claire Layering Guide",
            "url": "https://www.marieclaire.com/fashion/knitwear-layering-trends-2026/",
            "published_at": "2026-01-13T00:00:00Z",
        },
    },
    {
        "trend_id": "minimal_monochrome_2026",
        "label": "Minimal Monochrome",
        "scope": "global",
        "region": ["global"],
        "gender": ["male", "female", "unisex"],
        "categories": ["top", "bottom", "shoes"],
        "colors": ["black", "white", "charcoal", "grey"],
        "keywords": ["clean lines", "minimal", "structured tee", "tailored pant", "chelsea boot", "minimal sneaker"],
        "occasions": ["office", "smart_casual", "dinner", "night_out"],
        "season": ["spring", "summer", "autumn", "winter"],
        "strength": 0.80,
        "confidence": 0.86,
        "review_state": "unverified",
        "verification_status": "DEAD_LINK",
        "verification_note": "HTTP 404 on the claimed source URL (checked 2026-08-26).",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-12T08:00:00Z",
        "verified_at": None,
        "source": {
            "publisher": "GQ Minimalist Edit",
            "url": "https://www.gq.com/story/minimal-fashion-guide-2026",
            "published_at": "2026-01-10T00:00:00Z",
        },
    },
    {
        "trend_id": "modern_ethnic_2026",
        "label": "Contemporary Indian",
        "scope": "india",
        "region": ["india"],
        "gender": ["male", "female", "unisex"],
        "categories": ["top", "bottom", "ethnic_wear"],
        "colors": ["cream", "mustard", "maroon", "indigo", "terracotta", "white"],
        "keywords": ["short kurta", "linen kurta", "nehru jacket", "bandhgala", "indie fusion", "cotton tunic"],
        "occasions": ["festive", "family_event", "puja", "smart_casual", "wedding_guest"],
        "season": ["spring", "summer", "monsoon", "autumn", "winter"],
        "strength": 0.85,
        "confidence": 0.89,
        "review_state": "unverified",
        "verification_status": "DEAD_LINK",
        "verification_note": "HTTP 404 on the claimed source URL (checked 2026-08-26).",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-10T08:00:00Z",
        "verified_at": None,
        "source": {
            "publisher": "Vogue India Festive Report",
            "url": "https://www.vogue.in/fashion/content/indian-festive-ethnic-wear-trends-2026",
            "published_at": "2026-01-18T00:00:00Z",
        },
    },
    {
        "trend_id": "festive_jewel_tones_2026",
        "label": "Festive Jewel Tones",
        "scope": "india",
        "region": ["india"],
        "gender": ["male", "female", "unisex"],
        "categories": ["top", "bottom", "ethnic_wear", "dresses"],
        "colors": ["emerald", "ruby", "sapphire", "plum", "teal", "deep gold", "burgundy"],
        "keywords": ["silk", "brocade", "embroidery", "festive kurta", "sherwani", "lehenga", "anarkali"],
        "occasions": ["wedding", "festive", "reception", "diwali", "sangeet"],
        "season": ["autumn", "winter"],
        "strength": 0.90,
        "confidence": 0.92,
        "review_state": "unverified",
        "verification_status": "DEAD_LINK",
        "verification_note": "HTTP 404 on the claimed source URL (checked 2026-08-26).",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-09T08:00:00Z",
        "verified_at": None,
        "source": {
            "publisher": "Harper's Bazaar India Wedding Issue",
            "url": "https://www.harpersbazaar.in/fashion/jewel-tone-wedding-trends-2026",
            "published_at": "2026-01-07T00:00:00Z",
        },
    },
    {
        "trend_id": "statement_accessories_2026",
        "label": "Statement Accents",
        "scope": "global",
        "region": ["global"],
        "gender": ["male", "female", "unisex"],
        "categories": ["accessories", "jewelry", "bags", "footwear"],
        "colors": ["silver", "gold", "tan", "black", "metallic"],
        "keywords": ["chunky loafer", "leather tote", "silver chain", "minimal watch", "structured bag"],
        "occasions": ["smart_casual", "dinner", "office", "weekend"],
        "season": ["spring", "summer", "autumn", "winter"],
        "strength": 0.65,
        "confidence": 0.75,
        "review_state": "unverified",
        "verification_status": "DEAD_LINK",
        "verification_note": "HTTP 404 on the claimed source URL (checked 2026-08-26).",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "ingested_at": "2026-01-14T08:00:00Z",
        "verified_at": None,
        "source": {
            "publisher": "Elle Accessory Tracker",
            "url": "https://www.elle.com/fashion/accessories/statement-accents-2026",
            "published_at": "2026-01-12T00:00:00Z",
        },
    },
]


def get_trend_registry() -> List[Dict[str, Any]]:
    """Return all curated trend records (verified and unverified alike).

    Callers must go through TrendContextService.is_trend_valid()/
    get_active_trends() -- this function does not filter by verification
    status itself, it is the raw registry.
    """
    return ACTIVE_TRENDS
