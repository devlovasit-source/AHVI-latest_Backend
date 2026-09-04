#!/usr/bin/env python3
"""Cron / CLI utility to permanently hard-purge accounts whose 45-day grace period has expired.

Usage:
    python scripts/purge_expired_accounts.py [--dry-run] [--limit 50] [--user-id <uid>]

Scheduled execution (e.g. daily cron / Cloud Scheduler):
    0 3 * * * python /app/scripts/purge_expired_accounts.py >> /var/log/purge_expired.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

# Ensure backend root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.account_deletion_service import (
    execute_hard_purge,
    _parse_iso,
    ACCOUNT_STATUS_PENDING_DELETION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ahvi.scripts.purge_expired_accounts")


def find_expired_accounts(limit: int = 50) -> List[Dict[str, Any]]:
    """Finds all user profiles where account_status == 'pending_deletion' and deletion_scheduled_at <= now."""
    from services.appwrite_proxy import AppwriteProxy

    proxy = AppwriteProxy()
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    expired_accounts: List[Dict[str, Any]] = []

    # Attempt 1: Query using Appwrite queries if index is supported
    try:
        from appwrite.query import Query
        queries = [
            Query.equal("account_status", ACCOUNT_STATUS_PENDING_DELETION),
            Query.less_than_equal("deletion_scheduled_at", now_iso),
            Query.limit(limit),
        ]
        result = proxy.list_documents("users", limit=limit)
        # Note: AppwriteProxy handles query candidates; we additionally filter locally to guarantee precision
    except Exception as exc:
        logger.debug("Indexed query fallback to collection scan: %s", exc)

    # Attempt 2: Resilient full list with local timestamp evaluation
    try:
        page_limit = max(25, min(limit, 100))
        offset = 0
        while len(expired_accounts) < limit:
            docs_resp = proxy._list_documents_page(
                proxy._collection_id("users"),
                page_limit=page_limit,
                offset=offset,
            )
            docs = docs_resp.get("documents", [])
            if not docs:
                break

            for doc in docs:
                if not isinstance(doc, dict):
                    continue
                status = str(doc.get("account_status") or "").strip().lower()
                if status != ACCOUNT_STATUS_PENDING_DELETION:
                    continue

                scheduled_raw = doc.get("deletion_scheduled_at")
                if not scheduled_raw:
                    continue

                scheduled_dt = _parse_iso(str(scheduled_raw))
                if scheduled_dt and scheduled_dt <= now:
                    uid = str(doc.get("userId") or doc.get("user_id") or doc.get("$id") or "").strip()
                    if uid:
                        expired_accounts.append({
                            "user_id": uid,
                            "deletion_requested_at": doc.get("deletion_requested_at"),
                            "deletion_scheduled_at": scheduled_raw,
                            "deletion_reason": doc.get("deletion_reason"),
                        })
                        if len(expired_accounts) >= limit:
                            break

            if len(docs) < page_limit:
                break
            offset += len(docs)

    except Exception as exc:
        logger.exception("Error scanning user profiles for expired accounts: %s", exc)

    return expired_accounts


def main() -> int:
    parser = argparse.ArgumentParser(description="Purge accounts whose 45-day deletion period has passed.")
    parser.add_argument("--dry-run", action="store_true", help="Report expired accounts without deleting data.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of accounts to process.")
    parser.add_argument("--user-id", type=str, default=None, help="Execute immediate purge for a specific user ID.")
    args = parser.parse_args()

    logger.info("Starting purge_expired_accounts job (dry_run=%s)", args.dry_run)

    if args.user_id:
        target_uid = args.user_id.strip()
        logger.info("Targeted purge for single user_id=%s", target_uid)
        if args.dry_run:
            print(f"[DRY-RUN] Would execute hard purge for user_id: {target_uid}")
            return 0

        res = execute_hard_purge(target_uid)
        print(f"Purge result for {target_uid}: success={res.get('success')}")
        return 0 if res.get("success") else 1

    expired = find_expired_accounts(limit=args.limit)
    logger.info("Found %d expired accounts ready for hard purge", len(expired))

    if not expired:
        logger.info("No expired accounts found. Exiting.")
        return 0

    success_count = 0
    fail_count = 0

    for account in expired:
        uid = account["user_id"]
        sched = account.get("deletion_scheduled_at")
        if args.dry_run:
            logger.info("[DRY-RUN] Would purge user_id=%s (scheduled_at=%s)", uid, sched)
            continue

        try:
            logger.info("Executing hard purge for user_id=%s...", uid)
            result = execute_hard_purge(uid)
            if result.get("success"):
                success_count += 1
                logger.info("Hard purge succeeded user_id=%s", uid)
            else:
                fail_count += 1
                logger.warning("Hard purge reported partial/failed user_id=%s: %s", uid, result.get("targets"))
        except Exception as exc:
            fail_count += 1
            logger.exception("Unexpected exception executing hard purge user_id=%s: %s", uid, exc)

    logger.info(
        "Purge job completed. Processed=%d, Succeeded=%d, Failed=%d",
        len(expired),
        success_count,
        fail_count,
    )
    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
