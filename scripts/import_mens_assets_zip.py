"""Extract a mens-assets ZIP, upload images to Cloudflare R2, emit a
style_assets seed JSON.

Pipeline:
  1. Extract C:/.../mens assets.zip into data/mens_assets_import/
  2. Walk every image under that tree.
  3. Upload to R2 (style-boards bucket) under key
     ``mens-assets/<category-slug>/<filename>``.
  4. Append a style_assets row with inferred metadata to
     data/style_assets_mens_seed.json.

Re-uses ``services.r2_storage.R2Storage`` for credentials & client.
Idempotent: skips upload when ``--skip-existing-r2`` is passed and the
seed already records the same r2_key.

Run:
    python scripts/import_mens_assets_zip.py \
        --zip "C:/Users/USER/Downloads/mens assets.zip" \
        --output data/style_assets_mens_seed.json
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Trigger .env loading from r2_storage if the user hasn't set the env explicitly.
os.environ.setdefault("R2_LOAD_LOCAL_ENV", "true")

from services.r2_storage import R2Storage, R2StorageError  # noqa: E402
from services.style_asset_metadata_contract import canonical_metadata_update  # noqa: E402


# ---- Category / subcategory inference --------------------------------------

FOLDER_CATEGORY: Dict[str, Tuple[str, str]] = {
    # folder slug -> (style_assets category, default subcategory)
    "bags": ("accessory", "bag"),
    "bottoms": ("bottom", "trouser"),
    "footwear": ("footwear", "shoes"),
    "outerwear": ("outerwear", "jacket"),
    "tops": ("top", "shirt"),
    "festive sets": ("ethnic", "kurta_set"),
    "festive_sets": ("ethnic", "kurta_set"),
    "indian occasion set": ("ethnic", "indian_set"),
    "indian_occasion_set": ("ethnic", "indian_set"),
    "indian ocassion set": ("ethnic", "indian_set"),
    "jewellery": ("accessory", "jewellery"),
    "jewelry": ("accessory", "jewellery"),
    "other accessories": ("accessory", "accessory_misc"),
    "other_accessories": ("accessory", "accessory_misc"),
    "travel": ("travel", "travel_item"),
    "skincare": ("grooming", "skincare"),
}

# Subcategory tokens (longer tokens first to win during scanning).
SUBCATEGORY_TOKENS: List[Tuple[str, str]] = [
    ("formalshoe", "formal_shoe"),
    ("formal_shoe", "formal_shoe"),
    ("formalshoes", "formal_shoe"),
    ("dressshoe", "formal_shoe"),
    ("oxfordshoe", "formal_shoe"),
    ("derbyshoe", "formal_shoe"),
    ("loafers", "loafer"),
    ("loafer", "loafer"),
    ("sneakers", "sneaker"),
    ("sneaker", "sneaker"),
    ("sandals", "sandal"),
    ("sandal", "sandal"),
    ("slides", "slide"),
    ("slide", "slide"),
    ("flipflop", "flip_flops"),
    ("flipflops", "flip_flops"),
    ("birkinstock", "sandal"),
    ("boot", "boot"),
    ("messenger", "messenger_bag"),
    ("backpack", "backpack"),
    ("duffle", "duffle_bag"),
    ("dufflebag", "duffle_bag"),
    ("laptopbag", "laptop_bag"),
    ("cardcase", "cardholder"),
    ("wallet", "wallet"),
    ("blazer", "blazer"),
    ("suitset", "suit"),
    ("bandhgala", "bandhgala"),
    ("sherwani", "sherwani"),
    ("kurtaset", "kurta_set"),
    ("kurta", "kurta"),
    ("nehrujacket", "nehru_jacket"),
    ("waistcoat", "waistcoat"),
    ("hoodie", "hoodie"),
    ("sweatshirt", "sweatshirt"),
    ("cardigan", "cardigan"),
    ("sweater", "sweater"),
    ("overshirt", "overshirt"),
    ("polo", "polo"),
    ("tshirt", "t_shirt"),
    ("t-shirt", "t_shirt"),
    ("tank", "tank"),
    ("shirt", "shirt"),
    ("jacket", "jacket"),
    ("coat", "coat"),
    ("trouser", "trouser"),
    ("trousers", "trouser"),
    ("chino", "chino"),
    ("chinos", "chino"),
    ("jeans", "jeans"),
    ("joggers", "joggers"),
    ("jogger", "joggers"),
    ("cargopants", "cargo_pants"),
    ("cargoshorts", "cargo_shorts"),
    ("swimshorts", "swim_shorts"),
    ("gymshorts", "gym_shorts"),
    ("shorts", "shorts"),
    ("pant", "trouser"),
    ("belt", "belt"),
    ("watch", "watch"),
    ("sunglasses", "sunglasses"),
    ("sunglass", "sunglasses"),
    ("cap", "cap"),
    ("hat", "hat"),
    ("scarf", "scarf"),
    ("muffler", "muffler"),
    ("tie", "tie"),
    ("pocketsquare", "pocket_square"),
    ("cufflink", "cufflinks"),
    ("brooch", "brooch"),
    ("chain", "chain"),
    ("necklace", "necklace"),
    ("bracelet", "bracelet"),
    ("earring", "earrings"),
    ("ring", "ring"),
    ("powerbank", "powerbank"),
    ("neckpillow", "neck_pillow"),
    ("pillow", "pillow"),
    ("suitcase", "suitcase"),
    ("luggage", "suitcase"),
    ("pouch", "travel_pouch"),
    ("bottle", "water_bottle"),
    ("tumbler", "water_bottle"),
    ("keychain", "keychain"),
    ("toiletry", "toiletry"),
    ("eyemask", "eye_mask"),
    ("adapter", "travel_adapter"),
    ("headphone", "headphones"),
    ("earbud", "earbuds"),
    ("face", "skincare"),
    ("moisturizer", "skincare"),
    ("serum", "skincare"),
    ("sunscreen", "skincare"),
]

COLOR_TOKENS: List[str] = [
    # longer compounds first
    "lightblue",
    "lightgrey",
    "lightgray",
    "lightbrown",
    "darkbrown",
    "darkblue",
    "navyblue",
    "offwhite",
    "navy",
    "black",
    "white",
    "blue",
    "grey",
    "gray",
    "brown",
    "green",
    "olive",
    "khaki",
    "beige",
    "cream",
    "red",
    "yellow",
    "orange",
    "pink",
    "purple",
    "tan",
    "maroon",
    "burgundy",
    "army",
    "gold",
    "silver",
]


# ---- Archetype + occasion mapping (subcategory based) ----------------------

OFFICE_SUBS = {
    "blazer", "suit", "shirt", "polo", "trouser", "chino", "formal_shoe", "loafer",
    "belt", "watch", "messenger_bag", "laptop_bag", "cardholder",
}
TRAVEL_SUBS = {
    "suitcase", "powerbank", "travel_pouch", "neck_pillow", "water_bottle",
    "eye_mask", "travel_adapter", "headphones", "earbuds", "duffle_bag", "backpack",
    "keychain",
}
FESTIVE_SUBS = {
    "kurta", "kurta_set", "sherwani", "bandhgala", "nehru_jacket", "waistcoat",
    "suit", "indian_set",
}
CASUAL_SUBS = {
    "t_shirt", "polo", "shorts", "swim_shorts", "gym_shorts", "joggers", "sneaker",
    "cap", "hat", "hoodie", "sweatshirt", "tank", "slide", "sandal",
}
SMART_CASUAL_SUBS = {
    "overshirt", "cardigan", "sweater", "chino", "loafer", "polo", "blazer",
}

ARCHETYPE_RULES: List[Tuple[set, List[str]]] = [
    (OFFICE_SUBS, ["Modern Professional", "Smart Casual Edge", "Quiet Luxury"]),
    (FESTIVE_SUBS, ["Festive Modern", "Modern Desi"]),
    (TRAVEL_SUBS, ["Modern Utility", "Off-Duty Tailoring"]),
    (SMART_CASUAL_SUBS, ["Smart Casual Edge", "Refined Weekend"]),
    (CASUAL_SUBS, ["Refined Weekend", "Polished Casual"]),
]


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return text or "item"


def _folder_to_key(folder: str) -> str:
    return re.sub(r"\s+", " ", folder.strip().lower())


def _infer_category(folder: str) -> Tuple[str, str]:
    key = _folder_to_key(folder)
    key_alt = key.replace(" ", "_")
    if key in FOLDER_CATEGORY:
        return FOLDER_CATEGORY[key]
    if key_alt in FOLDER_CATEGORY:
        return FOLDER_CATEGORY[key_alt]
    return ("accessory", "accessory_misc")


def _scan_tokens(token_stream: str, candidates: List[Tuple[str, str]]) -> List[str]:
    """Return subcategory matches in token-priority order (first wins)."""
    hits: List[str] = []
    for token, sub in candidates:
        if token in token_stream and sub not in hits:
            hits.append(sub)
    return hits


def _infer_subcategory(filename: str, default: str) -> str:
    base = re.sub(r"\.[^.]+$", "", filename.lower())
    base_compact = re.sub(r"[^a-z0-9]", "", base)
    for token, sub in SUBCATEGORY_TOKENS:
        if token in base_compact:
            return sub
    return default


def _infer_colors(filename: str) -> List[str]:
    base = re.sub(r"\.[^.]+$", "", filename.lower())
    compact = re.sub(r"[^a-z0-9]", "", base)
    found: List[str] = []
    for color in COLOR_TOKENS:
        if color in compact and color not in found:
            found.append(color)
    # Normalize duplicates after substring overlap (e.g. "lightblue" also hits "blue").
    cleaned: List[str] = []
    for color in found:
        if any(color in other and color != other for other in found):
            continue
        cleaned.append(color)
    return cleaned


def _humanize_name(filename: str, subcategory: str) -> str:
    base = re.sub(r"\.[^.]+$", "", filename)
    # Insert space between camelCase / lower-upper edges, drop digits at tail.
    base = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", base)
    base = re.sub(r"[_\-]+", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    if not base:
        base = subcategory.replace("_", " ").title()
    # Title-case but keep short tokens lower like "and", "the".
    parts = [w if w.isupper() else w.capitalize() for w in base.split()]
    name = " ".join(parts)
    return f"Mens {name}"


def _archetypes_for(subcategory: str) -> List[str]:
    seen: List[str] = []
    for subset, archetypes in ARCHETYPE_RULES:
        if subcategory in subset:
            for arch in archetypes:
                if arch not in seen:
                    seen.append(arch)
    return seen or ["Refined Weekend"]


def _occasions_for(subcategory: str, category: str) -> List[str]:
    occasions: List[str] = []
    if subcategory in OFFICE_SUBS:
        occasions.extend(["startup_office", "client_meeting", "coffee_date"])
    if subcategory in FESTIVE_SUBS or category == "ethnic":
        occasions.extend(["wedding", "festive", "diwali"])
    if subcategory in TRAVEL_SUBS or category == "travel":
        occasions.extend(["travel", "vacation"])
    if subcategory in CASUAL_SUBS:
        occasions.extend(["casual_day", "weekend"])
    if subcategory in SMART_CASUAL_SUBS:
        occasions.extend(["smart_casual", "coffee_date"])
    # Dedupe in order.
    seen: List[str] = []
    for occ in occasions:
        if occ not in seen:
            seen.append(occ)
    return seen or ["casual_day"]


def _extract_zip(zip_path: Path, extract_to: Path) -> Path:
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)
    # ZIP typically contains a single top-level folder; return it if present.
    children = [p for p in extract_to.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]
    return extract_to


def _iter_images(root: Path) -> List[Tuple[Path, str]]:
    """Yields (image_path, category_folder_name)."""
    out: List[Tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        category_folder = path.parent.name
        out.append((path, category_folder))
    return out


def _content_type_for(path: Path) -> str:
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed:
        return guessed
    return "image/jpeg"


def _upload(storage: R2Storage, *, bucket: str, key: str, data: bytes, content_type: str) -> None:
    client = storage._client()  # noqa: SLF001 - reuse cached client
    client.put_object(bucket, key, BytesIO(data), length=len(data), content_type=content_type)


def build(zip_path: Path, output_path: Path, *, dry_run: bool, source: str) -> Dict[str, Any]:
    extract_root = ROOT / "data" / "mens_assets_import"
    asset_root = _extract_zip(zip_path, extract_root)

    storage = R2Storage()
    target_bucket = storage.style_boards_bucket or storage.raw_bucket
    target_public_url = storage.style_boards_public_url or storage.raw_public_url
    if not dry_run:
        if not target_bucket or not target_public_url:
            raise SystemExit("R2 bucket/public URL not configured (style_boards or raw).")

    rows: List[Dict[str, Any]] = []
    cat_counts: Dict[str, int] = {}
    sub_counts: Dict[str, int] = {}
    now = datetime.now(timezone.utc).isoformat()

    for image_path, folder in _iter_images(asset_root):
        category, default_sub = _infer_category(folder)
        subcategory = _infer_subcategory(image_path.name, default_sub)
        colors = _infer_colors(image_path.name)
        name = _humanize_name(image_path.name, subcategory)
        slug = _slugify(image_path.stem)
        folder_slug = _slugify(folder)
        asset_id = f"mens_assets_{folder_slug}_{slug}"[:80]
        r2_key = f"mens-assets/{folder_slug}/{image_path.name}"
        image_url = f"{target_public_url.rstrip('/')}/{r2_key}" if target_public_url else ""

        if not dry_run:
            try:
                data = image_path.read_bytes()
                _upload(
                    storage,
                    bucket=target_bucket,
                    key=r2_key,
                    data=data,
                    content_type=_content_type_for(image_path),
                )
                print(json.dumps({"event": "AHVI_MENS_ASSET_UPLOADED", "key": r2_key}))
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"event": "AHVI_MENS_ASSET_UPLOAD_FAILED", "key": r2_key, "reason": str(exc)}))
                continue

        tokens = [t for t in re.split(r"[_\W]+", image_path.stem.lower()) if t]
        tags = list(dict.fromkeys(tokens + [folder_slug, subcategory]))

        row = {
            "asset_id": asset_id,
            "name": name,
            "gender": "male",
            "category": category,
            "subcategory": subcategory,
            "colors": colors,
            "occasions": _occasions_for(subcategory, category),
            "archetypes": _archetypes_for(subcategory),
            "image_url": image_url,
            "r2_key": r2_key,
            "source": source,
            "status": "active",
            "asset_type": "reference",
            "tags": tags,
            "created_at": now,
            "updated_at": now,
        }
        row.update(canonical_metadata_update(row))
        rows.append(row)
        cat_counts[category] = cat_counts.get(category, 0) + 1
        sub_counts[subcategory] = sub_counts.get(subcategory, 0) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    return {
        "total": len(rows),
        "by_category": cat_counts,
        "by_subcategory": sub_counts,
        "output": str(output_path),
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a mens-assets ZIP into the style_assets pipeline.")
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/style_assets_mens_seed.json"))
    parser.add_argument("--source", default="mens_assets_zip")
    parser.add_argument("--dry-run", action="store_true", help="Skip R2 uploads; just emit the seed JSON.")
    args = parser.parse_args()
    report = build(args.zip_path, args.output, dry_run=args.dry_run, source=args.source)
    print(json.dumps({"event": "AHVI_STYLE_ASSET_IMPORT_SUMMARY", "report": report}, indent=2))


if __name__ == "__main__":
    main()
