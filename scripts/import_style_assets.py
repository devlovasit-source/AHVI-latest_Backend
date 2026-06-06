from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError


ALLOWED_FIELDS = {
    "asset_id",
    "name",
    "category",
    "subcategory",
    "source",
    "image_url",
    "r2_key",
    "colors",
    "archetypes",
    "occasions",
    "tags",
    "gender",
    "status",
    "created_at",
    "updated_at",
}


def _safe_doc_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]", "_", str(value or "").strip())
    return text[:36] or "unique()"


def _list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[,|]", text) if part.strip()]


def _normalize(row: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    asset_id = str(row.get("asset_id") or row.get("id") or row.get("name") or "").strip()
    payload = {
        "asset_id": asset_id,
        "name": str(row.get("name") or "").strip(),
        "category": str(row.get("category") or "").strip(),
        "subcategory": str(row.get("subcategory") or "").strip(),
        "source": str(row.get("source") or "").strip(),
        "image_url": str(row.get("image_url") or row.get("imageUrl") or "").strip(),
        "r2_key": str(row.get("r2_key") or row.get("r2Key") or "").strip(),
        "colors": _list(row.get("colors")),
        "archetypes": _list(row.get("archetypes")),
        "occasions": _list(row.get("occasions")),
        "tags": _list(row.get("tags")),
        "gender": str(row.get("gender") or "all").strip(),
        "status": str(row.get("status") or "active").strip(),
        "created_at": str(row.get("created_at") or row.get("createdAt") or now),
        "updated_at": now,
    }
    return {key: value for key, value in payload.items() if key in ALLOWED_FIELDS}


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("assets") or data.get("rows") or []
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list or an object with assets/rows.")
    return [dict(row) for row in data if isinstance(row, dict)]


def import_assets(path: Path, *, dry_run: bool = False) -> Dict[str, int]:
    proxy = AppwriteProxy()
    stats = {"scanned": 0, "created": 0, "updated": 0, "failed": 0}
    for row in _load_rows(path):
        stats["scanned"] += 1
        payload = _normalize(row)
        asset_id = str(payload.get("asset_id") or "").strip()
        if not asset_id or not payload.get("name") or not payload.get("image_url"):
            stats["failed"] += 1
            print(f"skip missing required fields: {row}")
            continue
        doc_id = _safe_doc_id(asset_id)
        if dry_run:
            print(f"dry-run upsert style_assets/{doc_id}: {payload['name']}")
            continue
        try:
            proxy.update_document("style_assets", doc_id, payload)
            stats["updated"] += 1
        except AppwriteProxyError:
            try:
                proxy.create_document("style_assets", payload, document_id=doc_id)
                stats["created"] += 1
            except Exception as exc:  # noqa: BLE001
                stats["failed"] += 1
                print(f"failed {asset_id}: {exc}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Import curated AHVI style assets.")
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stats = import_assets(args.json_path, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
