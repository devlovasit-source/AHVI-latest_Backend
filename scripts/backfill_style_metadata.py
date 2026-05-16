import argparse
import json
import logging
import os
import sys
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.appwrite_proxy import AppwriteProxy
from services.wardrobe_intelligence_service import enrich_wardrobe_item

log = logging.getLogger("ahvi.backfill_style_metadata")


def _doc_id(doc: Dict[str, Any]) -> str:
    return str(doc.get("$id") or doc.get("id") or doc.get("document_id") or "").strip()


def run(limit: int = 100, dry_run: bool = False) -> Dict[str, int]:
    proxy = AppwriteProxy()
    offset = 0
    updated = 0
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
            payload = {"style_metadata": json.dumps(enrich_wardrobe_item(doc))}
            if dry_run:
                log.info("dry_run doc=%s style_metadata=%s", doc_id, payload["style_metadata"])
                updated += 1
                continue
            try:
                proxy.update_document("outfits", doc_id, payload)
                updated += 1
            except Exception as exc:
                failed += 1
                log.warning("backfill failed doc=%s err=%s", doc_id, exc)

        meta = page.get("meta") if isinstance(page, dict) else {}
        if not meta.get("has_more") and len(docs) < limit:
            break
        offset += len(docs)

    return {"updated": updated, "skipped": skipped, "failed": failed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill style_metadata for wardrobe items.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(limit=args.limit, dry_run=args.dry_run), indent=2))
