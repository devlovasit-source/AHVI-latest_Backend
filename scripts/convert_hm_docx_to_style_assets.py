"""Convert H&M MENS image-URL .docx into a style_assets seed JSON.

Reads a Word document whose paragraphs are category headings followed by
H&M product image URLs (one per paragraph; sometimes several joined by
whitespace/newline). Emits a JSON list shaped for
``scripts.import_style_assets``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from docx import Document  # type: ignore
except ImportError as exc:  # pragma: no cover - import guarded
    raise SystemExit("python-docx required: pip install python-docx") from exc


HEADING_ALIASES: Dict[str, Tuple[str, str]] = {
    "T-SHIRTS": ("top", "t_shirt"),
    "T SHIRTS": ("top", "t_shirt"),
    "TSHIRTS": ("top", "t_shirt"),
    "SHIRTS": ("top", "shirt"),
    "JEANS": ("bottom", "jeans"),
    "SHORTS": ("bottom", "shorts"),
    "JACKETS & COATS": ("outerwear", "jacket"),
    "JACKETS&COATS": ("outerwear", "jacket"),
    "JACKETS AND COATS": ("outerwear", "jacket"),
    "HOODIES & SWEATSHIRTS": ("top", "hoodie_sweatshirt"),
    "HOODIES AND SWEATSHIRTS": ("top", "hoodie_sweatshirt"),
    "SLEEPWEAR & LOUNGEWEAR": ("loungewear", "sleepwear"),
    "SLEEPWEAR AND LOUNGEWEAR": ("loungewear", "sleepwear"),
    "SHOES": ("footwear", "shoes"),
    "FLIP FLOPS": ("footwear", "flip_flops"),
    "BRACELETS": ("accessory", "bracelet"),
    "NECKLACES": ("accessory", "necklace"),
    "RINGS": ("accessory", "ring"),
    "EARRINGS": ("accessory", "earrings"),
    "SUNGLASSES": ("accessory", "sunglasses"),
    "BELTS": ("accessory", "belt"),
    "HATS": ("accessory", "hat"),
}

# Section-only headings: don't pair with URLs directly.
SECTION_HEADINGS = {"MENS", "FOOTWARE", "FOOTWEAR", "ACCESSORIES"}

NAME_TEMPLATES: Dict[str, List[str]] = {
    "t_shirt": ["Essential T-Shirt", "Crew T-Shirt", "Relaxed T-Shirt", "Pocket T-Shirt", "Slim Fit T-Shirt"],
    "shirt": ["Oxford Shirt", "Relaxed Shirt", "Casual Shirt", "Linen Shirt", "Poplin Shirt"],
    "jeans": ["Dark Wash Jeans", "Straight Jeans", "Slim Jeans", "Tapered Jeans", "Light Wash Jeans"],
    "shorts": ["Linen Shorts", "Cargo Shorts", "Chino Shorts", "Drawstring Shorts", "Tailored Shorts"],
    "jacket": ["Lightweight Jacket", "Utility Jacket", "Bomber Jacket", "Overshirt", "Denim Jacket"],
    "hoodie_sweatshirt": [
        "Essential Hoodie",
        "Crew Sweatshirt",
        "Zip Hoodie",
        "Pullover Sweatshirt",
        "Oversized Hoodie",
    ],
    "sleepwear": ["Loungewear Set"],
    "shoes": ["Clean Sneakers", "Casual Loafers", "Lace-Up Boots", "Slip-On Sneakers", "Suede Boots"],
    "flip_flops": ["Flip Flops"],
    "bracelet": ["Bracelet"],
    "necklace": ["Necklace"],
    "ring": ["Ring"],
    "earrings": ["Earrings"],
    "sunglasses": ["Sunglasses"],
    "belt": ["Leather Belt"],
    "hat": ["Hat"],
}

OCCASION_MAP: Dict[str, List[str]] = {
    "t_shirt": ["coffee_date", "casual_day", "weekend"],
    "shirt": ["startup_office", "coffee_date", "smart_casual", "casual_day"],
    "jeans": ["coffee_date", "casual_day", "startup_office", "weekend"],
    "jacket": ["startup_office", "travel", "coffee_date", "casual_day"],
    "hoodie_sweatshirt": ["casual_day", "weekend", "travel"],
    "shoes": ["coffee_date", "startup_office", "casual_day", "date_night"],
    "belt": ["coffee_date", "startup_office", "casual_day", "travel"],
    "sunglasses": ["coffee_date", "startup_office", "casual_day", "travel"],
    "hat": ["coffee_date", "casual_day", "travel"],
    "bracelet": ["coffee_date", "startup_office", "casual_day", "date_night"],
    "ring": ["coffee_date", "startup_office", "casual_day", "date_night"],
    "necklace": ["coffee_date", "casual_day", "date_night"],
    "earrings": ["coffee_date", "casual_day", "date_night"],
    "shorts": ["beach", "vacation", "resort", "casual_day"],
    "flip_flops": ["beach", "vacation", "resort", "casual_day"],
    "sleepwear": ["loungewear"],
}

ARCHETYPE_MAP: Dict[str, List[str]] = {
    "t_shirt": ["Refined Weekend", "Smart Casual Edge", "Polished Casual"],
    "shirt": ["Modern Professional", "Refined Weekend", "Smart Casual Edge", "Contemporary Classic"],
    "jeans": ["Refined Weekend", "Off-Duty Tailoring", "Smart Casual Edge", "Modern Utility"],
    "jacket": ["Modern Utility", "Off-Duty Tailoring", "Urban Minimalist"],
    "hoodie_sweatshirt": ["Modern Utility", "Refined Weekend"],
    "shoes": ["Polished Casual", "Refined Weekend", "Smart Casual Edge"],
    "belt": ["Quiet Luxury", "Modern Professional", "Refined Weekend"],
    "sunglasses": ["Quiet Luxury", "Modern Professional", "Refined Weekend"],
    "hat": ["Refined Weekend", "Modern Utility"],
    "bracelet": ["Quiet Luxury", "Refined Weekend"],
    "ring": ["Quiet Luxury", "Refined Weekend"],
    "necklace": ["Quiet Luxury", "Refined Weekend"],
    "earrings": ["Quiet Luxury", "Refined Weekend"],
    "shorts": ["Italian Summer", "Resort Sophisticate", "Relaxed Weekend"],
    "flip_flops": ["Italian Summer", "Resort Sophisticate", "Relaxed Weekend"],
    "sleepwear": [],
}

# MVP gender rules for the H&M mens doc.
MALE_SUBCATS = {
    "t_shirt", "shirt", "jeans", "shorts", "jacket", "hoodie_sweatshirt",
    "shoes", "flip_flops", "bracelet", "ring", "sunglasses", "belt", "hat",
    "sleepwear",
}
UNISEX_SUBCATS = {"necklace"}
EXCLUDED_SUBCATS = {"earrings"}  # MVP: drop from mens seed
INACTIVE_SUBCATS = {"sleepwear"}  # imported but flagged inactive

URL_RE = re.compile(r"https?://\S+")


def _normalize_heading(text: str) -> str:
    """Strip leading numbering (e.g. ``1.SHOES``), trailing colons, and whitespace."""
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*\d+\s*[.)\-:]\s*", "", cleaned)
    cleaned = cleaned.rstrip(":").strip()
    return cleaned.upper()


def _is_url_paragraph(text: str) -> bool:
    return bool(URL_RE.search(text))


def _stable_asset_id(subcategory: str, url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"hm_mens_{subcategory}_{digest}"


def _build_name(subcategory: str, index: int) -> str:
    templates = NAME_TEMPLATES.get(subcategory, [subcategory.replace("_", " ").title()])
    base = templates[(index - 1) % len(templates)]
    return f"H&M Mens {base} {index:02d}"


def _resolve_gender(subcategory: str) -> str:
    if subcategory in UNISEX_SUBCATS:
        return "unisex"
    return "male"


def parse_document(paragraphs: List[str]) -> List[Dict[str, Any]]:
    """Walk paragraphs in order, tracking the latest heading.

    A paragraph is treated as a heading only when no URL is present in it.
    URL paragraphs are split on whitespace; every ``http(s)://`` token becomes
    a row under the current subcategory.
    """
    rows: List[Dict[str, Any]] = []
    current: Tuple[str, str] | None = None
    counters: Dict[str, int] = {}
    for raw in paragraphs:
        text = raw.strip()
        if not text:
            continue
        if not _is_url_paragraph(text):
            norm = _normalize_heading(text)
            if norm in SECTION_HEADINGS:
                current = None
                continue
            mapping = HEADING_ALIASES.get(norm)
            if mapping:
                current = mapping
            else:
                # Unknown heading: stop attributing URLs to a stale category.
                current = None
            continue
        if current is None:
            continue
        category, subcategory = current
        if subcategory in EXCLUDED_SUBCATS:
            continue
        for url in URL_RE.findall(text):
            counters[subcategory] = counters.get(subcategory, 0) + 1
            idx = counters[subcategory]
            rows.append(
                {
                    "asset_id": _stable_asset_id(subcategory, url),
                    "name": _build_name(subcategory, idx),
                    "gender": _resolve_gender(subcategory),
                    "category": category,
                    "subcategory": subcategory,
                    "colors": [],
                    "occasions": list(OCCASION_MAP.get(subcategory, [])),
                    "archetypes": list(ARCHETYPE_MAP.get(subcategory, [])),
                    "image_url": url,
                    "source": "H&M",
                    "status": "inactive" if subcategory in INACTIVE_SUBCATS else "active",
                    "asset_type": "reference",
                    "tags": [subcategory],
                }
            )
    return rows


def _read_paragraphs(path: Path) -> List[str]:
    document = Document(str(path))
    return [p.text for p in document.paragraphs]


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_subcat: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    by_gender: Dict[str, int] = {}
    for row in rows:
        by_subcat[row["subcategory"]] = by_subcat.get(row["subcategory"], 0) + 1
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        by_gender[row["gender"]] = by_gender.get(row["gender"], 0) + 1
    return {
        "total": len(rows),
        "by_subcategory": by_subcat,
        "by_status": by_status,
        "by_gender": by_gender,
    }


def convert(input_path: Path, output_path: Path) -> Dict[str, Any]:
    paragraphs = _read_paragraphs(input_path)
    rows = parse_document(paragraphs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return summarize(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert H&M mens docx to style_assets seed JSON.")
    parser.add_argument("--input", type=Path, required=True, help="Path to MENS.docx")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/style_assets_hm_mens_seed.json"),
        help="Seed JSON destination",
    )
    args = parser.parse_args()
    stats = convert(args.input, args.output)
    print(json.dumps({"event": "AHVI_STYLE_ASSET_IMPORT_DRY_RUN", "stats": stats, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
