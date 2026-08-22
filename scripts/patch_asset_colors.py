"""Backfill the `colors` field on style_assets that are missing it.

Targets ethnic/festive assets first (they logged AHVI_STYLE_ASSET_WEAK_METADATA
missing=['colors']). Infers colors from the asset name and updates the
document in place. Idempotent: skips assets that already have colors.

Run (dry run first):
    python scripts/patch_asset_colors.py --dry-run
    python scripts/patch_asset_colors.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("R2_LOAD_LOCAL_ENV", "true")

_COLOR_TOKENS = [
    "lightblue", "lightgreen", "lightbrown", "darkblue", "darkgreen",
    "navyblue", "offwhite", "navy", "black", "white", "blue", "grey", "gray",
    "brown", "brwon", "green", "olive", "khaki", "beige", "cream", "red",
    "yellow", "orange", "pink", "purple", "tan", "maroon", "marron",
    "burgundy", "burgendy", "gold", "silver", "lavender", "lilac", "peach",
    "chili", "rust", "wine", "teal",
]
# Misspelling / synonym canonicalization.
_COLOR_CANON = {
    "marron": "maroon",
    "brwon": "brown",
    "burgendy": "burgundy",
    "gray": "grey",
    "chili": "red",
}


def _infer_colors(*texts: Any) -> List[str]:
    """Infer colour tokens from asset text using word-boundary matching.

    A colour matches a whole word ("Tape red") or the start of a concatenated
    slug token ("blackbackpack", "yellowkurtaset", "brwonjeans") — but NOT a
    colour buried inside an unrelated word ("tailoRED", "tapeRED", "tailoREDshorts").
    Separators (_, -, /) are collapsed to spaces first so asset_id slug parts
    become matchable word boundaries.
    """
    blob = re.sub(r"[^a-z0-9]+", " ", " ".join(str(t or "") for t in texts).lower())
    found: List[str] = []
    for c in _COLOR_TOKENS:
        if re.search(r"\b" + re.escape(c), blob):
            canon = _COLOR_CANON.get(c, c)
            if canon not in found:
                found.append(canon)
    return [c for c in found if not any(c != o and c in o for o in found)]


def _has_colors(doc: Dict[str, Any]) -> bool:
    val = doc.get("colors")
    if isinstance(val, list):
        return any(str(v).strip() for v in val)
    return bool(str(val or "").strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ethnic-only", action="store_true", default=False)
    args = ap.parse_args()

    # Load .env into os.environ (the proxy reads os.environ directly).
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    from services.appwrite_proxy import AppwriteProxy

    proxy = AppwriteProxy()
    now = datetime.now(timezone.utc).isoformat()

    # Page through the whole collection.
    docs: List[Dict[str, Any]] = []
    offset = 0
    while True:
        page = proxy.list_documents("style_assets", limit=100, offset=offset, return_meta=True)
        rows = page.get("documents") if isinstance(page, dict) else page
        rows = [r for r in (rows or []) if isinstance(r, dict)]
        if not rows:
            break
        docs.extend(rows)
        offset += len(rows)
        if isinstance(page, dict):
            meta = page.get("meta") or {}
            if meta and not meta.get("has_more"):
                break
        if len(rows) < 100:
            break

    stats = {"total": len(docs), "candidates": 0, "patched": 0,
             "no_inference": 0, "already": 0, "errors": 0}

    for doc in docs:
        category = str(doc.get("category") or "").lower()
        if args.ethnic_only and category != "ethnic":
            continue
        if _has_colors(doc):
            stats["already"] += 1
            continue
        stats["candidates"] += 1
        colors = _infer_colors(doc.get("name"), doc.get("subcategory"), doc.get("asset_id"))
        if not colors:
            stats["no_inference"] += 1
            continue
        doc_id = str(doc.get("$id") or doc.get("id") or "")
        if not doc_id:
            continue
        if args.dry_run:
            stats["patched"] += 1
            continue
        try:
            proxy.update_document("style_assets", doc_id, {"colors": colors, "updated_at": now})
            stats["patched"] += 1
        except Exception as exc:  # noqa: BLE001
            print({"event": "PATCH_FAILED", "id": doc_id, "err": str(exc)[:140]})
            stats["errors"] += 1

    print({"event": "COLOR_PATCH_SUMMARY", "dry_run": args.dry_run, "stats": stats})


if __name__ == "__main__":
    main()
