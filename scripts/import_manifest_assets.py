"""Manifest-driven style-asset importer (men + women).

Reads the styling import manifest CSV, filters to fashion-safe P0/P1 rows,
extracts the matching images from a ZIP, uploads them to Cloudflare R2, and
upserts style_assets documents into Appwrite. Idempotent: dedupes by
asset_id (update if the document already exists, else create).

Unlike import_mens_assets_zip.py this trusts the manifest's gender/category/
sub_category/tags columns, so women ethnic (saree/lehenga/kurta set/gown)
are categorized correctly instead of falling back to "accessory".

Run (dry run first — no uploads, no writes, just counts):
    python scripts/import_manifest_assets.py \
        --manifest "C:/Users/USER/Downloads/styling_new_files_import_manifest.csv" \
        --zip "C:/Users/USER/Downloads/styling new files.zip" \
        --priorities P0 --dry-run

Then real:
    python scripts/import_manifest_assets.py \
        --manifest ... --zip ... --priorities P0 P1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import mimetypes
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("R2_LOAD_LOCAL_ENV", "true")

from services.style_asset_metadata_contract import canonical_metadata_update

# Fashion-safe filters ------------------------------------------------------

# Manifest category values that are never style assets.
_BLOCK_CATEGORIES = {"innerwear", "lifestyle"}
# Manifest sub_category values that are never style assets.
_BLOCK_SUBCATEGORIES = {"skincare", "travel", "bra"}

# Manifest category -> engine category vocabulary.
_CATEGORY_MAP = {
    "ethnic": "ethnic",
    "dresses": "dresses",
    "dress": "dresses",
    "tops": "top",
    "top": "top",
    "bottoms": "bottom",
    "bottom": "bottom",
    "footwear": "footwear",
    "outerwear": "outerwear",
    "accessories": "accessory",
    "accessory": "accessory",
}

# Occasion-ish tokens we lift out of the manifest tags into `occasions`.
_OCCASION_TOKENS = {
    "wedding", "festive", "reception", "occasion", "diwali", "sangeet",
    "haldi", "mehendi", "engagement", "party", "evening", "formal",
    "date", "casual", "cocktail", "office", "travel", "vacation", "brunch",
}

_COLOR_TOKENS = [
    "lightblue", "lightgreen", "lightbrown", "darkblue", "darkgreen",
    "navyblue", "offwhite", "navy", "black", "white", "blue", "grey", "gray",
    "brown", "green", "olive", "khaki", "beige", "cream", "red", "yellow",
    "orange", "pink", "purple", "tan", "maroon", "burgundy", "gold", "silver",
    "lavender", "lilac", "peach", "burntred", "barbiepink",
]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_") or "item"


def _infer_colors(name: str) -> List[str]:
    compact = re.sub(r"[^a-z0-9]", "", str(name or "").lower())
    found: List[str] = []
    for c in _COLOR_TOKENS:
        if c in compact and c not in found:
            found.append(c)
    # drop substrings of longer hits (blue when lightblue present)
    return [c for c in found if not any(c != o and c in o for o in found)]


def _split_tags(raw: str) -> Tuple[List[str], List[str]]:
    """Return (tags, occasions) from the manifest tag string."""
    parts = [t.strip().lower() for t in str(raw or "").split(",") if t.strip()]
    occasions = [t for t in parts if t in _OCCASION_TOKENS]
    return parts, occasions


def _eligible(row: Dict[str, str], priorities: set) -> Optional[str]:
    if str(row.get("include_style_assets", "")).strip().lower() != "true":
        return "not_included"
    if str(row.get("priority", "")).strip().upper() not in priorities:
        return "priority"
    if str(row.get("category", "")).strip().lower() in _BLOCK_CATEGORIES:
        return "blocked_category"
    if str(row.get("sub_category", "")).strip().lower() in _BLOCK_SUBCATEGORIES:
        return "blocked_subcategory"
    return None


def _row_to_payload(
    row: Dict[str, str], *, image_url: str, r2_key: str, now: str
) -> Dict[str, Any]:
    raw_gender = str(row.get("gender", "")).strip().lower()
    gender = "female" if raw_gender.startswith("w") else ("male" if raw_gender.startswith("m") else "unknown")
    folder = _slug(row.get("folder", ""))
    name_slug = _slug(row.get("name", ""))
    gender_prefix = "womens" if gender == "female" else ("mens" if gender == "male" else "style")
    asset_id = f"{gender_prefix}_assets_{folder}_{name_slug}"[:80]
    category = _CATEGORY_MAP.get(str(row.get("category", "")).strip().lower(), "")
    subcategory = _slug(row.get("sub_category", "")) or category
    tags, occasions = _split_tags(row.get("tags", ""))
    display = str(row.get("name", "")).strip() or subcategory.replace("_", " ").title()
    prefix = "Womens" if gender == "female" else "Mens"
    if not display.lower().startswith(prefix.lower()):
        display = f"{prefix} {display}"
    payload = {
        "asset_id": asset_id,
        "name": display,
        "gender": gender,
        "category": category,
        "subcategory": subcategory,
        "colors": _infer_colors(row.get("name", "")),
        "occasions": occasions,
        "archetypes": [],
        "image_url": image_url,
        "r2_key": r2_key,
        "source": "manifest_import",
        "status": "active",
        "asset_type": "reference",
        "tags": tags,
        "created_at": now,
        "updated_at": now,
    }
    payload.update(canonical_metadata_update(payload))
    return payload


def _build_zip_index(zip_path: Path) -> Dict[str, zipfile.ZipInfo]:
    """Map basename(lower) -> ZipInfo for fast lookup."""
    index: Dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            base = Path(info.filename).name.lower()
            if Path(base).suffix in {".jpg", ".jpeg", ".png", ".webp", ".jfif"}:
                index.setdefault(base, info)
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description="Manifest-driven style asset importer.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--zip", dest="zip_path", type=Path, required=True)
    ap.add_argument("--priorities", nargs="+", default=["P0", "P1"])
    ap.add_argument("--gender", choices=["women", "men", "all"], default="all")
    ap.add_argument(
        "--allow-update", action="store_true",
        help="Overwrite existing assets. Default is create-only (preserve existing).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    priorities = {p.upper() for p in args.priorities}
    want_gender = args.gender
    now = datetime.now(timezone.utc).isoformat()

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    skips: Dict[str, int] = {}
    eligible: List[Dict[str, str]] = []
    for row in rows:
        reason = _eligible(row, priorities)
        if not reason and want_gender != "all":
            g = "women" if str(row.get("gender", "")).strip().lower().startswith("w") else "men"
            if g != want_gender:
                reason = "gender_filter"
        if reason:
            skips[reason] = skips.get(reason, 0) + 1
        else:
            eligible.append(row)

    zip_index = _build_zip_index(args.zip_path)

    storage = None
    proxy = None
    public_url = ""
    bucket = ""
    if not args.dry_run:
        from services.r2_storage import R2Storage
        from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError

        storage = R2Storage()
        bucket = storage.style_boards_bucket or storage.raw_bucket
        public_url = (storage.style_boards_public_url or storage.raw_public_url or "").rstrip("/")
        proxy = AppwriteProxy()
        if not bucket or not public_url:
            raise SystemExit("R2 style_boards bucket/url not configured.")

    stats = {
        "eligible": len(eligible),
        "uploaded": 0, "created": 0, "updated": 0, "skipped_existing": 0,
        "image_missing": 0, "dup_asset_id": 0, "errors": 0,
        "women": 0, "men": 0, "ethnic": 0,
        "image_url_present": 0, "colors_present": 0,
    }
    seen_ids: set = set()

    for row in eligible:
        filename = Path(str(row.get("zip_path", ""))).name.lower()
        info = zip_index.get(filename)
        if info is None:
            stats["image_missing"] += 1
            continue

        gender = "female" if str(row.get("gender", "")).strip().lower().startswith("w") else "male"
        folder = _slug(row.get("folder", ""))
        r2_key = f"{'womens' if gender == 'female' else 'mens'}-assets/{folder}/{Path(info.filename).name}"
        image_url = f"{public_url}/{r2_key}" if public_url else f"r2://{r2_key}"

        payload = _row_to_payload(row, image_url=image_url, r2_key=r2_key, now=now)
        if payload["asset_id"] in seen_ids:
            stats["dup_asset_id"] += 1
            continue
        seen_ids.add(payload["asset_id"])

        # Appwrite documentId max 36 chars. Keep the full asset_id as a field;
        # use a deterministic short id for documents whose asset_id is too long
        # (the engine dedupes on the asset_id field, not $id).
        asset_id = payload["asset_id"]
        if len(asset_id) <= 36 and re.fullmatch(r"[A-Za-z0-9._-]+", asset_id):
            doc_id = asset_id
        else:
            doc_id = hashlib.sha1(asset_id.encode("utf-8")).hexdigest()[:32]

        stats["women" if gender == "female" else "men"] += 1
        if payload["category"] == "ethnic":
            stats["ethnic"] += 1
        if payload["image_url"]:
            stats["image_url_present"] += 1
        if payload["colors"]:
            stats["colors_present"] += 1

        if args.dry_run:
            continue

        # Create-only by default: if the document already exists, skip it so
        # existing assets (and their richer metadata) are preserved untouched.
        from services.appwrite_proxy import AppwriteProxyError
        if not args.allow_update:
            try:
                existing = proxy.get_document("style_assets", doc_id)
                if isinstance(existing, dict) and existing:
                    stats["skipped_existing"] += 1
                    continue
            except AppwriteProxyError:
                pass  # not found → proceed to create

        # Upload image to R2.
        try:
            with zipfile.ZipFile(args.zip_path) as zf:
                data = zf.read(info)
            client = storage._client()  # noqa: SLF001
            client.put_object(
                bucket, r2_key, BytesIO(data), length=len(data),
                content_type=mimetypes.guess_type(info.filename)[0] or "image/jpeg",
            )
            stats["uploaded"] += 1
        except Exception as exc:  # noqa: BLE001
            print({"event": "UPLOAD_FAILED", "key": r2_key, "err": str(exc)[:120]})
            stats["errors"] += 1
            continue

        try:
            try:
                proxy.update_document("style_assets", doc_id, payload)
                stats["updated"] += 1
            except AppwriteProxyError:
                proxy.create_document("style_assets", payload, document_id=doc_id)
                stats["created"] += 1
        except Exception as exc:  # noqa: BLE001
            print({"event": "APPWRITE_FAILED", "id": doc_id, "err": str(exc)[:160]})
            stats["errors"] += 1

    print({"event": "IMPORT_SUMMARY", "dry_run": args.dry_run,
           "skips": skips, "stats": stats})


if __name__ == "__main__":
    main()
