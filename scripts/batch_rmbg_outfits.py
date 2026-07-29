"""Re-cut opaque wardrobe images in the `outfits` collection.

Some outfits documents have masked_url / normalized_url pointing at a
catalog_<id>.png that is RGBA but FULLY OPAQUE (alpha 255 everywhere, baked
white background), so Wardrobe tiles and Style Boards render white rectangles.

This inspects the ACTUAL pixels of each item's masked_url (never the filename or
a status field), runs the existing RMBG service on the opaque ones, validates
the result, uploads a versioned cutout beside the original, and only then points
masked_url/normalized_url at it. image_url is left untouched as the original.

Dry-run by default; --apply writes. Reuses the validation helpers from
batch_rmbg_style_assets so there is one definition of "is this a real cutout".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests
import time
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.batch_rmbg_style_assets import (  # noqa: E402
    CUTOUT_VERSION,
    _encode_png,
    is_effectively_opaque,
    png_alpha_stats,
    validate_cutout_png,
)
from services.appwrite_proxy import AppwriteProxy  # noqa: E402
from services.bg_service import remove_bg_external_sync  # noqa: E402
from services.r2_storage import R2Storage  # noqa: E402

RESOURCE = "outfits"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _doc_id(doc: Dict[str, Any]) -> str:
    return _text(doc.get("$id") or doc.get("id"))


def _candidate_url(doc: Dict[str, Any]) -> str:
    """The image the app actually renders for this item."""
    return _text(doc.get("masked_url") or doc.get("normalized_url") or doc.get("image_url"))


def _cutout_key(doc: Dict[str, Any]) -> str:
    path = Path(urlparse(_candidate_url(doc)).path.lstrip("/"))
    stem = path.stem or f"outfit_{_doc_id(doc)}"
    parent = str(path.parent).replace("\\", "/")
    name = f"{stem}_cutout_v{CUTOUT_VERSION}.png"
    return name if parent in {"", "."} else f"{parent}/{name}"


# The CDN rate-limits rapid sequential fetches; without pacing+retry ~40 items
# came back as transient failures and were left UNCLASSIFIED (not "fine"), which
# made the candidate count unstable between runs.
DOWNLOAD_PACE_SECONDS = float(os.getenv("OUTFIT_RMBG_PACE_SECONDS", "0.35"))
DOWNLOAD_RETRIES = int(os.getenv("OUTFIT_RMBG_RETRIES", "4"))


def _download(url: str) -> bytes:
    """GET with exponential backoff. Raises only after the last attempt, so a
    transient 429/5xx never silently drops an item from the candidate set."""
    last: Exception | None = None
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"http_{resp.status_code}")
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < DOWNLOAD_RETRIES - 1:
                time.sleep(min(8.0, (2 ** attempt) * 0.75))
    raise last if last else RuntimeError("download_failed")


def _list_outfits(scan_limit: int) -> List[Dict[str, Any]]:
    proxy = AppwriteProxy()
    rows: List[Dict[str, Any]] = []
    offset = 0
    while offset < scan_limit:
        batch = proxy.list_documents(RESOURCE, limit=min(100, scan_limit - offset), offset=offset)
        if isinstance(batch, dict):
            batch = batch.get("documents") or []
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < 100:
            break
    return rows


def _preflight(apply: bool) -> None:
    if os.getenv("RMBG_SERVICE_URL", "").strip():
        return
    msg = "RMBG_SERVICE_URL is required. bg_service fails OPEN and would republish the opaque image."
    if apply:
        print(msg, file=sys.stderr)
        sys.exit(2)
    print(f"[WARN] {msg} (dry-run)", file=sys.stderr)


def _upload_png(storage: R2Storage, *, key: str, data: bytes) -> str:
    bucket = storage.wardrobe_bucket
    base = storage.wardrobe_public_url
    if not bucket or not base:
        raise RuntimeError("Missing wardrobe R2 bucket/public URL configuration.")
    storage._client().put_object(bucket, key, BytesIO(data), length=len(data), content_type="image/png")
    return f"{base.rstrip('/')}/{key}"


def process(*, apply: bool, scan_limit: int, limit: int, doc_id: str, snapshot_path: str = "") -> Dict[str, Any]:
    _preflight(apply)
    proxy = AppwriteProxy()
    storage = R2Storage()
    rows = _list_outfits(scan_limit)
    if doc_id:
        rows = [r for r in rows if _doc_id(r) == doc_id]
    stats = {"scanned": 0, "already_transparent": 0, "processed": 0, "updated": 0, "failed": 0, "skipped": 0}
    # Rollback snapshot: the pre-change URLs for every opaque candidate. Written
    # outside the repo (contains user ids + private wardrobe URLs).
    snapshot: List[Dict[str, Any]] = []

    for doc in rows:
        did = _doc_id(doc)
        url = _candidate_url(doc)
        stats["scanned"] += 1
        if not url:
            stats["skipped"] += 1
            continue
        try:
            source = _download(url)
            time.sleep(DOWNLOAD_PACE_SECONDS)
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            print(json.dumps({"event": "AHVI_OUTFIT_RMBG_FAILED", "doc_id": did, "reason": f"download:{str(exc)[:60]}"}))
            continue

        _has_alpha, ratio, reason = png_alpha_stats(source)
        if reason.startswith("invalid_image"):
            stats["failed"] += 1
            print(json.dumps({"event": "AHVI_OUTFIT_RMBG_FAILED", "doc_id": did, "reason": reason}))
            continue
        if not is_effectively_opaque(source):
            stats["already_transparent"] += 1
            print(json.dumps({"event": "AHVI_OUTFIT_RMBG_SKIP", "doc_id": did, "reason": "already_transparent"}))
            continue

        stats["processed"] += 1
        entry = {
            "doc_id": did,
            "user_id": _text(doc.get("userId") or doc.get("user_id")),
            "name": _text(doc.get("name")),
            "category": _text(doc.get("category")),
            "image_url": _text(doc.get("image_url")),
            "masked_url": _text(doc.get("masked_url")),
            "normalized_url": _text(doc.get("normalized_url")),
            "replacement_url": "",
            "source_transparent_ratio": round(ratio, 4),
        }
        snapshot.append(entry)
        print(json.dumps({"event": "AHVI_OUTFIT_RMBG_START", "doc_id": did}))
        try:
            png = _encode_png(remove_bg_external_sync(source))
            ok, why, out_ratio = validate_cutout_png(png)
            if not ok:
                raise RuntimeError(why)
            key = _cutout_key(doc)
            public_url = f"dry-run://{key}"
            entry["replacement_url"] = public_url
            if apply:
                public_url = _upload_png(storage, key=key, data=png)
                entry["replacement_url"] = public_url
                # image_url stays as the original; only the rendered fields move.
                proxy.update_document(RESOURCE, did, {"masked_url": public_url, "normalized_url": public_url})
                stats["updated"] += 1
                print(json.dumps({"event": "AHVI_OUTFIT_RMBG_UPDATE", "doc_id": did, "field": "masked_url"}))
            print(json.dumps({"event": "AHVI_OUTFIT_RMBG_OK", "doc_id": did, "transparent_ratio": round(out_ratio, 4)}))
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            print(json.dumps({"event": "AHVI_OUTFIT_RMBG_FAILED", "doc_id": did, "reason": str(exc)[:120]}))
        if limit and stats["processed"] >= limit:
            break
    if snapshot_path:
        Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
        Path(snapshot_path).write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"event": "AHVI_OUTFIT_SNAPSHOT_WRITTEN", "count": len(snapshot), "path": snapshot_path}))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-cut opaque wardrobe images in the outfits collection.")
    ap.add_argument("--apply", action="store_true", help="Write R2/Appwrite updates. Default is dry-run.")
    ap.add_argument("--scan-limit", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="Cap processed (opaque) items.")
    ap.add_argument("--doc-id", default="", help="Process a single outfits document id.")
    ap.add_argument("--snapshot", default="", help="Write a rollback snapshot JSON to this path (keep it outside the repo).")
    args = ap.parse_args()
    stats = process(
        apply=bool(args.apply),
        scan_limit=max(1, args.scan_limit),
        limit=max(0, args.limit),
        doc_id=args.doc_id,
        snapshot_path=args.snapshot,
    )
    print(json.dumps({"success": True, "dry_run": not args.apply, "stats": stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
