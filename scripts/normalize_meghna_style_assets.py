"""Normalize Meghna-curated style assets into a style_assets seed JSON.

Accepts either a single JSON file or a directory of JSONs (e.g. the
``asset_library/`` and ``capsule wardrobes/`` exports). Each input is
expected to be a JSON object with an ``assets`` list, an object with a
``rows`` list, or a bare list of asset dicts. Items are normalized into
the shape consumed by ``scripts.import_style_assets``.

Assets missing ``image_url`` AND ``asset_path`` are logged and rejected.
URLs are never synthesized.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CATEGORY_ALIASES: Dict[str, Tuple[str, str | None]] = {
    "tops": ("top", None),
    "top": ("top", None),
    "tshirt": ("top", "t_shirt"),
    "t-shirt": ("top", "t_shirt"),
    "shirt": ("top", "shirt"),
    "bottoms": ("bottom", None),
    "bottom": ("bottom", None),
    "jeans": ("bottom", "jeans"),
    "trousers": ("bottom", "trousers"),
    "shorts": ("bottom", "shorts"),
    "outerwear": ("outerwear", None),
    "jackets": ("outerwear", "jacket"),
    "coats": ("outerwear", "coat"),
    "footwear": ("footwear", None),
    "shoes": ("footwear", "shoes"),
    "sandals": ("footwear", "sandals"),
    "loungewear": ("loungewear", None),
    "sleepwear": ("loungewear", "sleepwear"),
    "accessories": ("accessory", None),
    "accessory": ("accessory", None),
    "bags": ("accessory", "bag"),
    "bag": ("accessory", "bag"),
    "phone slings": ("accessory", "phone_sling"),
    "phone_sling": ("accessory", "phone_sling"),
    "travel pouches": ("accessory", "travel_pouch"),
    "water bottles": ("accessory", "water_bottle"),
    "keychains": ("accessory", "keychain"),
    "sets": ("set", None),
    "festive sets": ("set", "festive"),
    "earrings": ("accessory", "earrings"),
    "necklaces": ("accessory", "necklace"),
    "bracelets": ("accessory", "bracelet"),
    "rings": ("accessory", "ring"),
    "sunglasses": ("accessory", "sunglasses"),
    "belts": ("accessory", "belt"),
    "hats": ("accessory", "hat"),
}

GENDER_ALIASES: Dict[str, str] = {
    "men": "male",
    "mens": "male",
    "male": "male",
    "women": "female",
    "womens": "female",
    "female": "female",
    "unisex": "unisex",
    "all": "unisex",
}

PERSONA_TO_ARCHETYPE: Dict[str, str] = {
    "clean_minimal": "Urban Minimalist",
    "preppy_smart": "Modern Professional",
    "sporty_ease": "Modern Utility",
    "modern_desi": "Festive Modern",
    "boho_artisanal": "Boho Artisanal",
    "glam_luxe": "Quiet Luxury",
    "soft_elegant": "Refined Weekend",
    "street_cool": "Smart Casual Edge",
}


def _to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[,|]", text) if part.strip()]


def _normalize_category(raw_category: Any, fallback: str | None) -> Tuple[str, str]:
    """Returns (category, subcategory). Empty strings if undeterminable."""
    candidates: List[str] = []
    for value in (raw_category, fallback):
        if value:
            candidates.append(str(value).strip().lower())
    for cand in candidates:
        mapping = CATEGORY_ALIASES.get(cand)
        if mapping:
            cat, sub = mapping
            return cat, sub or ""
    # Best-effort: return raw category as-is.
    if candidates:
        return candidates[0], ""
    return "", ""


def _normalize_gender(raw: Any) -> str:
    if isinstance(raw, list):
        normalized = {GENDER_ALIASES.get(str(v).strip().lower(), "") for v in raw}
        normalized.discard("")
        if not normalized:
            return ""
        if normalized == {"male"}:
            return "male"
        if normalized == {"female"}:
            return "female"
        return "unisex"
    if raw is None:
        return ""
    return GENDER_ALIASES.get(str(raw).strip().lower(), "")


def _archetypes_from_personas(personas: Iterable[str]) -> List[str]:
    seen: List[str] = []
    for persona in personas:
        mapped = PERSONA_TO_ARCHETYPE.get(str(persona).strip().lower())
        if mapped and mapped not in seen:
            seen.append(mapped)
    return seen


def _pick_image(item: Dict[str, Any]) -> Tuple[str, str]:
    """Returns (image_url, asset_path). Either can be empty."""
    for key in ("image_url", "imageUrl", "url"):
        val = item.get(key)
        if val and str(val).strip().lower().startswith(("http://", "https://")):
            return str(val).strip(), ""
    refs = item.get("image_refs") or []
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, str) and ref.lower().startswith(("http://", "https://")):
                return ref.strip(), ""
            if isinstance(ref, str) and ref.strip():
                return "", ref.strip()
    for key in ("asset_path", "assetPath", "path", "local_path"):
        val = item.get(key)
        if val:
            return "", str(val).strip()
    return "", ""


def _iter_items_from_payload(payload: Any) -> Iterable[Tuple[Dict[str, Any], str | None]]:
    """Yields (item_dict, fallback_category) for each asset in a payload."""
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item, None
        return
    if not isinstance(payload, dict):
        return
    fallback_cat = payload.get("category") or payload.get("categoryName")
    container = payload.get("assets") or payload.get("rows") or payload.get("items")
    if isinstance(container, list):
        for item in container:
            if isinstance(item, dict):
                yield item, fallback_cat


def _load_json(path: Path) -> Any:
    """Read a JSON file tolerating the leading ``{{`` typo seen in some inputs."""
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r"^\s*\{\{", "{", raw, count=1)
        return json.loads(fixed)


def _normalize_item(item: Dict[str, Any], fallback_category: str | None) -> Tuple[Dict[str, Any] | None, str]:
    asset_id = str(item.get("asset_id") or item.get("id") or item.get("$id") or "").strip()
    name = str(item.get("name") or item.get("title") or "").strip()
    raw_subcat = item.get("subcategory") or item.get("subCategory")
    raw_category = item.get("category") or item.get("categoryName") or raw_subcat
    category, subcategory = _normalize_category(raw_category, fallback_category)
    if not subcategory and raw_subcat:
        subcategory = str(raw_subcat).strip().lower().replace("-", "_")
    gender = _normalize_gender(item.get("gender") or item.get("gender_support") or item.get("genderSupport"))
    image_url, asset_path = _pick_image(item)
    archetypes = _to_list(item.get("archetypes")) or _archetypes_from_personas(_to_list(item.get("personas")))
    row = {
        "asset_id": asset_id or re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "",
        "name": name,
        "gender": gender or "unisex",
        "category": category,
        "subcategory": subcategory,
        "colors": _to_list(item.get("colors")),
        "occasions": _to_list(item.get("occasions")),
        "archetypes": archetypes,
        "image_url": image_url,
        "asset_path": asset_path,
        "source": "Meghna",
        "status": str(item.get("status") or "active").strip().lower(),
        "asset_type": str(item.get("asset_type") or "reference").strip().lower(),
        "tags": _to_list(item.get("tags")),
    }
    missing: List[str] = []
    if not row["name"]:
        missing.append("name")
    if not row["category"]:
        missing.append("category")
    if not row["gender"]:
        missing.append("gender")
    if not row["image_url"] and not row["asset_path"]:
        missing.append("image_url_or_asset_path")
    if missing:
        return None, ",".join(missing)
    return row, ""


def _walk_inputs(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(p for p in path.rglob("*.json") if p.is_file())
    return [path]


def normalize(input_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []
    files = _walk_inputs(input_path)
    for file_path in files:
        try:
            payload = _load_json(file_path)
        except Exception as exc:  # noqa: BLE001
            rejected.append({"file": str(file_path), "reason": f"parse_error:{exc}"})
            continue
        for item, fallback_cat in _iter_items_from_payload(payload):
            row, reason = _normalize_item(item, fallback_cat)
            if row is None:
                rejected.append(
                    {"file": str(file_path), "asset_id": str(item.get("asset_id") or item.get("id") or ""), "reason": reason}
                )
                continue
            rows.append(row)
    by_category: Dict[str, int] = {}
    by_gender: Dict[str, int] = {}
    missing_image = 0
    for row in rows:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1
        by_gender[row["gender"]] = by_gender.get(row["gender"], 0) + 1
        if not row["image_url"]:
            missing_image += 1
    summary = {
        "total": len(rows),
        "rejected": len(rejected),
        "missing_image_url": missing_image,
        "by_category": by_category,
        "by_gender": by_gender,
        "files_scanned": len(files),
    }
    return rows, {"summary": summary, "rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Meghna style assets to style_assets seed JSON.")
    parser.add_argument("--input", type=Path, required=True, help="JSON file or directory of JSON files")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/style_assets_meghna_seed.json"),
        help="Seed JSON destination",
    )
    args = parser.parse_args()
    rows, report = normalize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"event": "AHVI_STYLE_ASSET_IMPORT_DRY_RUN", "report": report, "output": str(args.output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
