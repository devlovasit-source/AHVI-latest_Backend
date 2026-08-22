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
from services.agent_metadata_validator import (
    metadata_v2_coverage,
    normalize_metadata_v2,
    upsert_wardrobe_style_metadata_sync,
    validate_metadata_payload,
    validate_wardrobe_metadata_sync,
)
from services.wardrobe_intelligence_service import enrich_wardrobe_item

log = logging.getLogger("ahvi.backfill_style_metadata")
STYLE_METADATA_RESOURCE = "wardrobe_style_metadata"
_SAFE_DOC_ID_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_NEW_STYLE_KEYS = {
    "metadata_version",
    "formality_score",
    "shine_level",
    "statement_level_score",
    "visual_noise_score",
    "best_for",
    "avoid_for",
    "style_keywords",
    "fabric_feel",
    "silhouette",
    "visual_noise",
    "pattern_intensity",
    "statement_level",
    "professionalism_score",
    "client_meeting_score",
    "boardroom_score",
    "capsule_score",
    "versatility_score",
    "date_night_score",
    "risk_flags",
    "styling_notes",
}


def _doc_id(doc: Dict[str, Any]) -> str:
    return str(doc.get("$id") or doc.get("id") or doc.get("document_id") or "").strip()


def _safe_document_id(value: Any) -> str:
    safe = _SAFE_DOC_ID_RE.sub("_", str(value or "")).strip("._-")
    return (safe or "metadata")[:36]


def _merge_if_present(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in source.items():
        if value in (None, "", [], {}):
            continue
        target[key] = value
    return target


def _style_metadata_for_doc(
    doc: Dict[str, Any],
    *,
    user_id: str,
    use_agent: bool,
) -> Dict[str, Any]:
    """Build refreshed metadata for a wardrobe row.

    The legacy local enrichment keeps existing AHVI heuristics. The validator
    pass adds the new professional/capsule/date scores and category-aware
    defaults even when the external validator agent is not used.
    """

    metadata = enrich_wardrobe_item(doc if isinstance(doc, dict) else {})
    if not isinstance(metadata, dict):
        metadata = {}

    deterministic = validate_metadata_payload({}, base_item=doc)
    _merge_if_present(metadata, deterministic)

    if use_agent:
        agent_meta = validate_wardrobe_metadata_sync(
            item=doc,
            user_id=user_id,
            vision_result=doc.get("vision_result") if isinstance(doc, dict) else None,
            context={"source": "backfill_style_metadata"},
        )
        if isinstance(agent_meta, dict):
            _merge_if_present(metadata, agent_meta)
            metadata["agent_validated"] = True
            metadata["agent_confidence"] = float(agent_meta.get("confidence") or 0.0)

    return normalize_metadata_v2(metadata, base_item=doc)


def _metadata_payload(
    doc_id: str,
    user_id: str,
    doc: Dict[str, Any],
    *,
    use_agent: bool,
) -> Dict[str, Any]:
    style_metadata = _style_metadata_for_doc(doc, user_id=user_id, use_agent=use_agent)
    return {
        "item_id": doc_id,
        "userId": user_id,
        "style_metadata": json.dumps(
            style_metadata, separators=(",", ":"), ensure_ascii=False
        ),
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


def _has_new_style_keys(doc: Dict[str, Any]) -> bool:
    raw = doc.get("style_metadata")
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    if not isinstance(raw, dict):
        return False
    return bool(_NEW_STYLE_KEYS.intersection(raw.keys()))


def run(
    *,
    user_id: str = "",
    all_users: bool = False,
    limit: int = 100,
    dry_run: bool = False,
    force: bool = False,
    use_agent: bool = False,
) -> Dict[str, int]:
    if not all_users and not user_id:
        raise ValueError("Pass --user-id for a scoped backfill, or --all-users.")

    target_user_id = user_id
    proxy = AppwriteProxy()
    offset = 0
    updated = 0
    created = 0
    skipped = 0
    failed = 0
    scanned = 0
    unchanged = 0
    coverage_items = []

    while True:
        page = proxy.list_documents(
            "outfits",
            user_id=(target_user_id or None),
            limit=limit,
            offset=offset,
            return_meta=True,
        )
        docs = page.get("documents") if isinstance(page, dict) else page
        if not isinstance(docs, list) or not docs:
            break

        for doc in docs:
            scanned += 1
            if not isinstance(doc, dict):
                skipped += 1
                continue
            coverage_items.append(doc)
            doc_id = _doc_id(doc)
            if not doc_id:
                skipped += 1
                continue
            doc_user_id = str(doc.get("userId") or doc.get("user_id") or "").strip()
            if not doc_user_id:
                skipped += 1
                continue
            if not force and _has_new_style_keys(doc):
                unchanged += 1
                continue

            payload = _metadata_payload(doc_id, doc_user_id, doc, use_agent=use_agent)
            if dry_run:
                style_meta = json.loads(payload.get("style_metadata") or "{}")
                log.info(
                    "AHVI_METADATA_V2_DRY_RUN doc=%s user=%s category=%s subcategory=%s formality=%s formality_score=%.2f shine=%.2f statement=%.2f avoid_for=%s risk_flags=%s",
                    doc_id,
                    doc_user_id,
                    style_meta.get("category"),
                    style_meta.get("subcategory"),
                    style_meta.get("formality"),
                    float(style_meta.get("formality_score") or 0.0),
                    float(style_meta.get("shine_level") or 0.0),
                    float(style_meta.get("statement_level_score") or 0.0),
                    style_meta.get("avoid_for") or [],
                    style_meta.get("risk_flags") or [],
                )
                updated += 1
                continue
            try:
                style_meta = json.loads(payload.get("style_metadata") or "{}")
                result_doc = upsert_wardrobe_style_metadata_sync(
                    doc_user_id, doc_id, style_meta
                )
                result = str(result_doc.get("status") or "")
                if result == "failed":
                    raise RuntimeError(result_doc.get("reason") or "metadata upsert failed")
                if result == "created":
                    created += 1
                else:
                    updated += 1
                log.info(
                    "AHVI_METADATA_V2_MIGRATED doc=%s user=%s status=%s",
                    doc_id,
                    doc_user_id,
                    result,
                )
            except Exception as exc:
                failed += 1
                log.warning("backfill failed doc=%s err=%s", doc_id, exc)

        meta = page.get("meta") if isinstance(page, dict) else {}
        if not meta.get("has_more") and len(docs) < limit:
            break
        offset += len(docs)

    coverage = metadata_v2_coverage(coverage_items)
    log.info("AHVI_METADATA_V2_AUDIT coverage=%s", coverage)
    return {
        "scanned": scanned,
        "updated": updated,
        "created": created,
        "unchanged": unchanged,
        "skipped": skipped,
        "failed": failed,
        **coverage,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill style_metadata for wardrobe items.")
    parser.add_argument("--user-id", default="", help="Backfill one user's wardrobe.")
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="Backfill every wardrobe row. Prefer --user-id for release fixes.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Refresh rows even if metadata already has new score keys.")
    parser.add_argument("--use-agent", action="store_true", help="Call the external metadata validator agent when enabled.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(
        json.dumps(
            run(
                user_id=args.user_id,
                all_users=args.all_users,
                limit=args.limit,
                dry_run=args.dry_run,
                force=args.force,
                use_agent=args.use_agent,
            ),
            indent=2,
        )
    )
