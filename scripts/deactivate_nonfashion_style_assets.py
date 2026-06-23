"""Deactivate non-fashion style_assets (status=inactive). DRY-RUN by default.

Reads data/style_asset_hygiene_proposal.csv and selects rows where
is_fashion_asset == false (no) AND confidence == high. Sets status=inactive via
the existing AppwriteProxy.update_document path. Never deletes documents, never
touches other metadata. Aborts unless exactly 24 targets (override with
--allow-count-mismatch).

    python scripts/deactivate_nonfashion_style_assets.py                      # dry run
    python scripts/deactivate_nonfashion_style_assets.py --apply              # write
    python scripts/deactivate_nonfashion_style_assets.py --allow-count-mismatch
"""
from __future__ import annotations
import os, sys, csv, argparse
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("R2_LOAD_LOCAL_ENV", "true")
_envp = ROOT / ".env"
if _envp.exists():
    for line in _envp.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("="); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

HYGIENE = ROOT / "data" / "style_asset_hygiene_proposal.csv"
BACKUP = ROOT / "data" / "nonfashion_deactivate_backup.csv"
REPORT = ROOT / "data" / "nonfashion_deactivate_report.csv"
EXPECT = 24
FALSE_VALS = {"false", "no", "0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--allow-count-mismatch", action="store_true")
    args = ap.parse_args()
    apply = bool(args.apply)
    mode = "APPLY" if apply else "DRY RUN"

    if not HYGIENE.exists():
        raise SystemExit("Missing %s — run catalog_hygiene_audit.py first." % HYGIENE)
    with open(HYGIENE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    targets = [r for r in rows
               if str(r.get("is_fashion_asset", "")).strip().lower() in FALSE_VALS
               and str(r.get("confidence", "")).strip().lower() == "high"]

    print("=== deactivate_nonfashion — %s ===" % mode)
    print("hygiene rows: %d | selected (non-fashion & high): %d" % (len(rows), len(targets)))
    if len(targets) != EXPECT and not args.allow_count_mismatch:
        raise SystemExit("ABORT: expected exactly %d targets, found %d. "
                         "Re-run with --allow-count-mismatch to override." % (EXPECT, len(targets)))

    # map asset_id -> live doc (read-only)
    from services.appwrite_proxy import AppwriteProxy
    p = AppwriteProxy()
    docs, off = [], 0
    while True:
        pg = p.list_documents("style_assets", limit=100, offset=off, return_meta=True)
        ds = [d for d in (pg.get("documents") or []) if isinstance(d, dict)]
        if not ds: break
        docs += ds; off += len(ds)
        if (pg.get("meta") or {}).get("has_more") == False: break
        if len(ds) < 100: break
    by_aid = {str(d.get("asset_id") or ""): d for d in docs}

    now = datetime.now(timezone.utc).isoformat()
    report, backup = [], []
    stats = {"selected": len(targets), "planned": 0, "deactivated": 0,
             "skipped_no_doc": 0, "already_inactive": 0, "failed": 0}

    for r in targets:
        aid = r["asset_id"]; d = by_aid.get(aid)
        if not d:
            stats["skipped_no_doc"] += 1
            report.append(dict(asset_id=aid, action="skip", reason="no live doc", old_status="", status="skipped"))
            continue
        old_status = str(d.get("status") or "")
        did = str(d.get("$id") or d.get("id") or "")
        if old_status.lower() == "inactive":
            stats["already_inactive"] += 1
            report.append(dict(asset_id=aid, action="skip", reason="already inactive", old_status=old_status, status="skipped"))
            continue
        stats["planned"] += 1
        if not apply:
            report.append(dict(asset_id=aid, action="would_deactivate", reason=r.get("predicted_type", ""), old_status=old_status, status="planned"))
            continue
        backup.append(dict(asset_id=aid, document_id=did, old_status=old_status, predicted_type=r.get("predicted_type", "")))
        try:
            # only status changes; all other metadata preserved
            p.update_document("style_assets", did, {"status": "inactive", "updated_at": now})
            stats["deactivated"] += 1
            report.append(dict(asset_id=aid, action="deactivated", reason=r.get("predicted_type", ""), old_status=old_status, status="ok"))
        except Exception as e:
            stats["failed"] += 1
            report.append(dict(asset_id=aid, action="deactivate", reason=str(e)[:120], old_status=old_status, status="failed"))

    with open(REPORT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["asset_id", "action", "reason", "old_status", "status"], extrasaction="ignore")
        w.writeheader(); w.writerows(report)
    if apply:
        with open(BACKUP, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["asset_id", "document_id", "old_status", "predicted_type"])
            w.writeheader(); w.writerows(backup)

    print("report ->", REPORT, "| backup ->", BACKUP if apply else "(written only on --apply)")
    print("stats:", stats)
    if not apply:
        print("DRY RUN — no DB writes. %d would be set status=inactive. Re-run with --apply to write." % stats["planned"])


if __name__ == "__main__":
    main()
