"""Apply HIGH-confidence colour proposals to style_assets.

DRY-RUN by default. Applies ONLY rows with confidence=high AND status=ok from
data/style_asset_color_image_proposal.csv. Skips medium/low. Cross-guards
against the hygiene audit (never colours a non-fashion / flagged asset).
Uses the existing AppwriteProxy.update_document path. Writes a report always
and a rollback backup before any write.

    python scripts/apply_color_proposal.py            # dry run (default) — no writes
    python scripts/apply_color_proposal.py --apply    # write high-conf colours
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

PROPOSAL = ROOT / "data" / "style_asset_color_image_proposal.csv"
HYGIENE = ROOT / "data" / "style_asset_hygiene_proposal.csv"
REPORT = ROOT / "data" / "color_apply_report.csv"
BACKUP = ROOT / "data" / "color_apply_backup.csv"


def _read_csv(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write to DB (default: dry run)")
    args = ap.parse_args()
    apply = bool(args.apply)
    mode = "APPLY" if apply else "DRY RUN"

    proposals = _read_csv(PROPOSAL)
    if not proposals:
        raise SystemExit("No proposal CSV at %s — run extract_style_asset_colors_dryrun.py first." % PROPOSAL)

    # hygiene cross-guard: exclude anything not is_fashion_asset==yes
    nonfashion = {r["asset_id"] for r in _read_csv(HYGIENE) if r.get("is_fashion_asset") != "yes"}

    # HIGH + ok only
    high = [r for r in proposals
            if r.get("confidence") == "high" and r.get("status") == "ok" and r.get("suggested_colors")]
    targets = [r for r in high if r["asset_id"] not in nonfashion]
    excluded = [r for r in high if r["asset_id"] in nonfashion]

    # map asset_id -> ($id, current colors) from live DB (read-only)
    from services.appwrite_proxy import AppwriteProxy
    p = AppwriteProxy()
    docs, off = [], 0
    while True:
        pg = p.list_documents("style_assets", limit=100, offset=off, return_meta=True)
        rows = [r for r in (pg.get("documents") or []) if isinstance(r, dict)]
        if not rows: break
        docs += rows; off += len(rows)
        if (pg.get("meta") or {}).get("has_more") == False: break
        if len(rows) < 100: break
    by_aid = {str(d.get("asset_id") or ""): d for d in docs}

    now = datetime.now(timezone.utc).isoformat()
    report, backup = [], []
    stats = {"high_total": len(high), "excluded_nonfashion": len(excluded),
             "planned": 0, "applied": 0, "skipped_no_doc": 0, "skipped_has_colors": 0, "failed": 0}

    for r in targets:
        aid = r["asset_id"]; proposed = [c for c in r["suggested_colors"].split("|") if c]
        d = by_aid.get(aid)
        if not d:
            stats["skipped_no_doc"] += 1
            report.append(dict(asset_id=aid, action="skip", reason="no live doc", proposed="|".join(proposed), current="", status="skipped"))
            continue
        cur = d.get("colors") or []
        if isinstance(cur, str): cur = [cur] if cur else []
        if cur:  # never overwrite existing colours
            stats["skipped_has_colors"] += 1
            report.append(dict(asset_id=aid, action="skip", reason="already has colors", proposed="|".join(proposed), current="|".join(cur), status="skipped"))
            continue
        did = str(d.get("$id") or d.get("id") or "")
        stats["planned"] += 1
        if not apply:
            report.append(dict(asset_id=aid, action="would_update", reason="high-conf", proposed="|".join(proposed), current="", status="planned"))
            continue
        backup.append(dict(asset_id=aid, document_id=did, old_colors="|".join(cur)))
        try:
            p.update_document("style_assets", did, {"colors": proposed, "updated_at": now})
            stats["applied"] += 1
            report.append(dict(asset_id=aid, action="updated", reason="high-conf", proposed="|".join(proposed), current="", status="ok"))
        except Exception as e:
            stats["failed"] += 1
            report.append(dict(asset_id=aid, action="update", reason=str(e)[:120], proposed="|".join(proposed), current="", status="failed"))

    # always write report; write backup only when applying
    with open(REPORT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["asset_id", "action", "reason", "proposed", "current", "status"], extrasaction="ignore")
        w.writeheader(); w.writerows(report)
    if apply:
        with open(BACKUP, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["asset_id", "document_id", "old_colors"]); w.writeheader(); w.writerows(backup)

    print("=== apply_color_proposal — %s ===" % mode)
    print("report ->", REPORT, "| backup ->", BACKUP if apply else "(not written in dry run)")
    print("stats:", stats)
    if excluded:
        print("excluded by hygiene (non-fashion, NOT coloured):", [r["asset_id"] for r in excluded][:10])
    if not apply:
        print("DRY RUN — no DB writes. Re-run with --apply to write the planned %d." % stats["planned"])


if __name__ == "__main__":
    main()
