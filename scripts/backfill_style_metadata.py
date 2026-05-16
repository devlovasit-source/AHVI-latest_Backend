import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError
from services.wardrobe_intelligence_service import enrich_wardrobe_item

log = logging.getLogger("ahvi.backfill_style_metadata")
STYLE_METADATA_RESOURCE = "wardrobe_style_metadata"
_SAFE_DOC_ID_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _doc_id(doc: Dict[str, Any]) -> str:
    return str(doc.get("$id") or doc.get("id") or doc.get("document_id") or "").strip()


def _safe_document_id(value: Any) -> str:
    safe = _SAFE_DOC_ID_RE.sub("_", str(value or "")).strip("._-")
    return (safe or "metadata")[:36]


def _metadata_payload(doc_id: str, user_id: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "item_id": doc_id,
        "userId": user_id,
        "style_metadata": json.dumps(enrich_wardrobe_item(doc)),
    }


def _upsert_style_metadata(proxy: AppwriteProxy, doc_id: str, payload: Dict[str, Any]) -> str:
    metadata_id = _safe_document_id(doc_id)
    try:
        proxy.update_document(STYLE_METADATA_RESOURCE, metadata_id, payload)
        return "updated"
    except AppwriteProxyError as exc:
        if "404" not in str(exc):
            raise
    proxy.create_document(STYLE_METADATA_RESOURCE, payload, document_id=metadata_id)
    return "created"


def run(limit: int = 100, dry_run: bool = False) -> Dict[str, int]:
    proxy = AppwriteProxy()
    offset = 0
    updated = 0
    created = 0
    skipped = 0
    failed = 0

    while True:
        page = proxy.list_documents("outfits", limit=limit, offset=offset, return_meta=True)
        docs = page.get("documents") if isinstance(page, dict) else page
        if not isinstance(docs, list) or not docs:
            break

        for doc in docs:
            if not isinstance(doc, dict):
                skipped += 1
                continue
            doc_id = _doc_id(doc)
            if not doc_id:
                skipped += 1
                continue
            user_id = str(doc.get("userId") or doc.get("user_id") or "").strip()
            if not user_id:
                skipped += 1
                continue
            payload = _metadata_payload(doc_id, user_id, doc)
            if dry_run:
                log.info(
                    "dry_run metadata doc=%s payload=%s",
                    doc_id,
                    json.dumps(payload, ensure_ascii=False),
                )
                updated += 1
                continue
            try:
                result = _upsert_style_metadata(proxy, doc_id, payload)
                if result == "created":
                    created += 1
                else:
                    updated += 1
            except Exception as exc:
                failed += 1
                log.warning("backfill failed doc=%s err=%s", doc_id, exc)

        meta = page.get("meta") if isinstance(page, dict) else {}
        if not meta.get("has_more") and len(docs) < limit:
            break
        offset += len(docs)

    return {"updated": updated, "created": created, "skipped": skipped, "failed": failed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill style_metadata for wardrobe items.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(limit=args.limit, dry_run=args.dry_run), indent=2))
