"""Audit and optionally backfill canonical style-asset metadata.

Read-only is the default. Writes require both ``--apply`` and the explicit
``--confirm-apply`` safety flag. No LLM or network enrichment is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.appwrite_proxy import AppwriteProxy
from services.style_asset_metadata_contract import (
    canonical_metadata_update,
    normalize_style_asset_metadata,
    summarize_style_asset_metadata,
)


def _documents(proxy: Any, *, limit: int, asset_id: str = "") -> List[Dict[str, Any]]:
    if asset_id:
        try:
            row = proxy.get_document("style_assets", asset_id)
        except Exception:  # noqa: BLE001 - audit reports an empty match safely
            # ``asset_id`` is usually the document id, but legacy imports may
            # have a different Appwrite $id. Fall back to a bounded scan.
            offset = 0
            while offset < max(1, int(limit)):
                page_limit = min(100, max(1, int(limit)) - offset)
                page = proxy.list_documents(
                    "style_assets", limit=page_limit, offset=offset, return_meta=True
                )
                page_rows = page.get("documents", []) if isinstance(page, dict) else page
                page_rows = [item for item in (page_rows or []) if isinstance(item, dict)]
                match = next(
                    (item for item in page_rows if str(item.get("asset_id") or "") == asset_id),
                    None,
                )
                if match:
                    return [match]
                if not page_rows:
                    break
                offset += len(page_rows)
                meta = page.get("meta", {}) if isinstance(page, dict) else {}
                if meta and not meta.get("has_more"):
                    break
            return []
        return [row] if isinstance(row, dict) else []

    rows: List[Dict[str, Any]] = []
    offset = 0
    target = max(1, int(limit))
    while len(rows) < target:
        page_limit = min(100, target - len(rows))
        page = proxy.list_documents(
            "style_assets", limit=page_limit, offset=offset, return_meta=True
        )
        page_rows = page.get("documents", []) if isinstance(page, dict) else page
        page_rows = [row for row in (page_rows or []) if isinstance(row, dict)]
        if not page_rows:
            break
        rows.extend(page_rows)
        offset += len(page_rows)
        meta = page.get("meta", {}) if isinstance(page, dict) else {}
        if (meta and not meta.get("has_more")) or (not meta and len(page_rows) < page_limit):
            break
    return rows[:target]


def build_audit(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    source_rows = list(rows)
    normalized = [
        normalize_style_asset_metadata(row, trusted_style_asset_source=True)
        for row in source_rows
    ]
    proposed: List[Dict[str, Any]] = []
    for raw, asset in zip(source_rows, normalized):
        canonical = canonical_metadata_update(raw)
        changes = {key: value for key, value in canonical.items() if raw.get(key) != value}
        if changes:
            proposed.append({
                "asset_id": asset.get("asset_id"),
                "document_id": raw.get("$id") or raw.get("id") or asset.get("asset_id"),
                "metadata_status": asset.get("metadata_status"),
                "changes": changes,
            })
    return {
        "summary": summarize_style_asset_metadata(normalized),
        "assets": [
            {
                "asset_id": asset.get("asset_id"),
                "metadata_status": asset.get("metadata_status"),
                "metadata_score": asset.get("metadata_score"),
                "role": asset.get("role"),
                "gender_fit": asset.get("gender_fit"),
                "missing_metadata_fields": asset.get("missing_metadata_fields"),
            }
            for asset in normalized
        ],
        "proposed_updates": proposed,
        "proposed_update_count": len(proposed),
        "applied_count": 0,
    }


def audit_style_assets(
    *,
    proxy: Any | None = None,
    limit: int = 1000,
    asset_id: str = "",
    apply: bool = False,
    confirm_apply: bool = False,
) -> Dict[str, Any]:
    if apply and not confirm_apply:
        raise ValueError("--apply requires --confirm-apply")
    client = proxy or AppwriteProxy()
    report = build_audit(_documents(client, limit=limit, asset_id=asset_id))
    if apply:
        updated_at = datetime.now(timezone.utc).isoformat()
        for proposal in report["proposed_updates"]:
            changes = dict(proposal["changes"])
            changes["metadata_updated_at"] = updated_at
            client.update_document("style_assets", proposal["document_id"], changes)
            report["applied_count"] += 1
    return report


def _write_csv(path: Path, assets: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "asset_id", "metadata_status", "metadata_score", "role", "gender_fit",
        "missing_metadata_fields",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for asset in assets:
            row = dict(asset)
            row["missing_metadata_fields"] = "|".join(row.get("missing_metadata_fields") or [])
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit canonical Style asset metadata readiness.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--asset-id", default="")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-apply", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.confirm_apply:
        parser.error("--apply requires --confirm-apply")

    report = audit_style_assets(
        limit=args.limit,
        asset_id=args.asset_id,
        apply=args.apply,
        confirm_apply=args.confirm_apply,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(args.csv, report["assets"])
    print(rendered)


if __name__ == "__main__":
    main()
