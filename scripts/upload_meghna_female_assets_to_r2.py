"""Upload Meghna female style-asset images to Cloudflare R2.

Reads data/style_assets_meghna_r2_manifest.json, pulls each source image out
of `womens assets.zip`, converts the ones flagged needs_conversion=true to
JPEG (RGB), and uploads to the EXACT target_r2_key so the URLs already baked
into data/style_assets_meghna_seed.json resolve.

Re-uses services.r2_storage.R2Storage for credentials/client (style-boards
bucket), matching scripts/import_mens_assets_zip.py.

SAFETY:
  * Default mode is DRY RUN. Nothing uploads unless you pass --apply.
  * Item-level failures do not abort the batch (only under --apply).
  * This script does NOT import style_assets, touch Appwrite, or commit.

Usage:
    # plan only (default) — no upload:
    python scripts/upload_meghna_female_assets_to_r2.py
    python scripts/upload_meghna_female_assets_to_r2.py --dry-run

    # actually upload (explicit opt-in):
    python scripts/upload_meghna_female_assets_to_r2.py --apply
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Let services.r2_storage load .env locally (same as the mens importer).
os.environ.setdefault("R2_LOAD_LOCAL_ENV", "true")

MANIFEST = ROOT / "data" / "style_assets_meghna_r2_manifest.json"
RESULT_OUT = ROOT / "data" / "style_assets_meghna_r2_upload_result.json"
# womens assets.zip is the curated source set (kept outside the repo).
DEFAULT_ZIP = Path(r"C:\Users\USER\Downloads\womens assets.zip")

JPEG_QUALITY = 90


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _load_manifest() -> List[Dict[str, Any]]:
    if not MANIFEST.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST}")
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit("Manifest is empty or not a list.")
    return rows


def _validate(rows: List[Dict[str, Any]], zf: zipfile.ZipFile,
              public_base: str) -> List[str]:
    """Returns a list of fatal validation errors (empty = ok)."""
    errors: List[str] = []
    names = set(zf.namelist())

    # 3a. no duplicate target_r2_key
    dup = {k: c for k, c in Counter(r.get("target_r2_key") for r in rows).items() if c > 1}
    if dup:
        errors.append(f"duplicate target_r2_key in manifest: {dup}")

    # 3b. each source exists in the zip
    missing = [r["source_path_inside_zip"] for r in rows
               if not r.get("source_path_inside_zip") or r["source_path_inside_zip"] not in names]
    if missing:
        errors.append(f"{len(missing)} source path(s) not found in zip; first few: {missing[:5]}")

    # 3c. target_public_url == public_base + '/' + target_r2_key (internal consistency)
    if public_base:
        bad_url = [r["asset_id"] for r in rows
                   if r.get("target_public_url") != f"{public_base.rstrip('/')}/{r['target_r2_key']}"]
        if bad_url:
            errors.append(
                f"{len(bad_url)} row(s) where target_public_url != configured public base + key. "
                f"Configured base={public_base!r}. First: {bad_url[:5]}. "
                "Uploading would NOT resolve at the seed URLs — fix config or manifest first."
            )
    return errors


# ---------------------------------------------------------------------------
# Bytes prep (conversion)
# ---------------------------------------------------------------------------
def _prepare_bytes(raw: bytes, needs_conversion: bool) -> bytes:
    """Return upload-ready bytes. Convert to JPEG/RGB when flagged."""
    if not needs_conversion:
        return raw
    from PIL import Image  # local import so --dry-run works without Pillow
    im = Image.open(io.BytesIO(raw))
    if im.mode not in ("RGB",):
        im = im.convert("RGB")
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Upload Meghna female assets to R2 (dry-run by default).")
    ap.add_argument("--apply", action="store_true",
                    help="Actually upload. Without this flag the script only plans (dry run).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Explicit dry run (default behaviour when --apply is absent).")
    ap.add_argument("--zip", default=str(DEFAULT_ZIP), help="Path to womens assets.zip")
    args = ap.parse_args()

    apply = bool(args.apply)  # default False -> dry run
    mode = "APPLY (uploading)" if apply else "DRY RUN (no upload)"

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"Source zip not found: {zip_path}")

    rows = _load_manifest()
    zf = zipfile.ZipFile(zip_path)

    # Resolve R2 target (style-boards bucket, matching the mens importer).
    storage = None
    target_bucket = ""
    public_base = ""
    if apply:
        from services.r2_storage import R2Storage, R2StorageError  # noqa: F401
        storage = R2Storage()
        target_bucket = storage.style_boards_bucket or storage.raw_bucket
        public_base = storage.style_boards_public_url or storage.raw_public_url
        if not target_bucket or not public_base:
            raise SystemExit("R2 style_boards/raw bucket + public URL not configured.")
    else:
        # In dry run, read the public base from env if present so the URL
        # consistency check still runs; otherwise derive it from the manifest.
        public_base = (
            os.getenv("R2_URL_STYLE_BOARDS")
            or os.getenv("EXPO_PUBLIC_R2_URL_STYLE_BOARDS")
            or ""
        )
        if not public_base and rows:
            # derive base from first row (everything before '/female-assets/')
            u = rows[0].get("target_public_url", "")
            public_base = u.split("/female-assets/")[0] if "/female-assets/" in u else ""

    errors = _validate(rows, zf, public_base)
    print(f"=== Meghna female R2 upload — {mode} ===")
    print(f"manifest rows : {len(rows)}")
    print(f"source zip    : {zip_path}")
    print(f"target bucket : {target_bucket or '(resolved only under --apply)'}")
    print(f"public base   : {public_base or '(unknown)'}")
    if errors:
        print("\nFATAL VALIDATION ERRORS — aborting (no uploads):")
        for e in errors:
            print("  - " + e)
        raise SystemExit(1)
    print("validation    : OK (rows, sources, no dup keys, url<->key consistent)")

    stats = Counter()
    results: List[Dict[str, Any]] = []
    client = storage._client() if apply else None  # noqa: SLF001 (mirrors mens importer)

    for r in rows:
        stats["total"] += 1
        aid = r["asset_id"]
        src = r["source_path_inside_zip"]
        key = r["target_r2_key"]
        needs_conv = str(r.get("needs_conversion", "false")).lower() == "true"
        rec: Dict[str, Any] = {"asset_id": aid, "target_r2_key": key,
                               "needs_conversion": needs_conv, "status": "", "error": ""}
        try:
            if not apply:
                action = "CONVERT+UPLOAD" if needs_conv else "UPLOAD"
                print(f"  [plan] {action:14} {src}  ->  {key}")
                if needs_conv:
                    stats["converted"] += 1
                stats["planned"] += 1
                rec["status"] = "planned"
                results.append(rec)
                continue

            raw = zf.read(src)
            data = _prepare_bytes(raw, needs_conv)
            if needs_conv:
                stats["converted"] += 1
            client.put_object(
                target_bucket, key, io.BytesIO(data),
                length=len(data), content_type="image/jpeg",
            )
            stats["uploaded"] += 1
            rec["status"] = "uploaded"
            rec["bytes"] = len(data)
            print(f"  [ok]   {key}")
        except Exception as exc:  # noqa: BLE001 — item-level isolation
            stats["failed"] += 1
            rec["status"] = "failed"
            rec["error"] = str(exc)[:200]
            print(f"  [FAIL] {key} :: {str(exc)[:160]}")
        results.append(rec)

    print("\n=== SUMMARY ===")
    print(f"  total     : {stats['total']}")
    print(f"  uploaded  : {stats['uploaded']}")
    print(f"  converted : {stats['converted']}")
    print(f"  planned   : {stats['planned']} (dry run only)")
    print(f"  skipped   : {stats['skipped']}")
    print(f"  failed    : {stats['failed']}")

    if apply:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bucket": target_bucket,
            "public_base": public_base,
            "summary": dict(stats),
            "results": results,
        }
        RESULT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nresult report -> {RESULT_OUT}")
    else:
        print("\nDry run only — no files uploaded, no report written. Re-run with --apply to upload.")


if __name__ == "__main__":
    main()
