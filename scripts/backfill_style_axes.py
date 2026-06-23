"""P0 metadata backfill — formality / movement / energy / aesthetic / avoid_for.

Rule-based, deterministic, ADDITIVE (never overwrites an existing non-empty
value), idempotent. Derives the 5 style-intelligence axes the catalog is missing
from each asset's name/subcategory/category/tags blob.

Goal: give the scorer real garment intelligence so e.g. a music festival board
never picks oxford + belt + loafers.

Usage:
  python scripts/backfill_style_axes.py                 # dry-run summary
  python scripts/backfill_style_axes.py --apply         # write back (+ .bak)
  python scripts/backfill_style_axes.py --file data/style_assets_mens_seed.json --apply

Live catalog is in Appwrite — this edits the SEED so re-imports carry the axes.
A separate patch script is needed to push axes onto live DB rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

DEFAULT_FILE = "data/style_assets_meghna_seed.json"

# Occasion tokens used by _asset_score avoid_for (-12) and occasion_style_rules.
# Broad spellings so the veto fires whatever canonical token reaches scoring.
_FESTIVAL = ["music festival", "music_festival", "festival", "concert", "rave", "gig", "edm"]
_FORMAL_OCC = ["gym", "workout", "beach", "music festival", "concert", "rave"]
_CASUAL_OCC = ["office", "wedding", "funeral", "temple", "interview", "conference"]


def _blob(a: dict) -> str:
    parts = [a.get("name"), a.get("subcategory"), a.get("category")]
    parts += list(a.get("tags") or [])
    return " ".join(str(p) for p in parts if p).lower()


def _derive(a: dict) -> dict:
    """Return the 5 axes for one asset. formality 1-5, movement/energy 1-9."""
    b = _blob(a)
    cat = str(a.get("category") or "").lower()

    def has(*words):
        return any(w in b for w in words)

    # ---- defaults (neutral mid) ----
    formality, movement, energy = 3, 5, 4
    aesthetic = "casual"
    avoid_for: list[str] = []

    # ---- high-formality / low-movement classics ----
    if has("oxford", "derby", "brogue", "dress shoe", "monk strap", "suit", "tuxedo", "blazer", "sport coat"):
        formality, movement, energy, aesthetic = 5, 3, 3, "formal"
        avoid_for = list(_FESTIVAL) + ["gym", "beach"]
    elif has("loafer"):
        formality, movement, energy, aesthetic = 4, 4, 3, "classic"
        avoid_for = list(_FESTIVAL) + ["gym"]
    elif has("heel", "stiletto", "pump", "gown"):
        formality, movement, energy, aesthetic = 5, 2, 4, "formal"
        avoid_for = ["gym", "airport", "beach"]
    # ---- ethnic ----
    elif cat == "ethnic" or has("kurta", "sherwani", "bandhgala", "nehru", "mojari", "jutti",
                                 "lehenga", "saree", "sari", "dupatta", "churidar", "anarkali", "achkan"):
        formality, movement, energy, aesthetic = 4, 4, 6, "ethnic"
        avoid_for = ["gym", "office", "airport", "beach", "workout"]
    # ---- street / movement-first ----
    elif has("graphic tee", "graphic", "sneaker", "trainer", "cargo", "jogger", "track", "hoodie", "sweatshirt"):
        formality, movement, energy, aesthetic = 2, 8, 7, "street"
        avoid_for = ["wedding", "funeral", "interview"]
    elif has("tee", "t-shirt", "tshirt", "tank"):
        formality, movement, energy, aesthetic = 1, 7, 6, "casual"
        avoid_for = ["wedding", "funeral", "office", "interview"]
    elif has("polo"):
        formality, movement, energy, aesthetic = 3, 6, 5, "smart_casual"
    elif has("overshirt", "camp collar", "linen shirt"):
        formality, movement, energy, aesthetic = 2, 6, 5, "relaxed"
    # ---- bottoms ----
    elif has("formal trouser", "tailored trouser", "dress pant"):
        formality, movement, energy, aesthetic = 5, 3, 3, "formal"
        avoid_for = list(_FESTIVAL) + ["gym", "beach"]
    elif has("trouser", "chino"):
        formality, movement, energy, aesthetic = 3, 5, 4, "smart_casual"
    elif has("cargo pant", "cargo"):
        formality, movement, energy, aesthetic = 2, 9, 6, "street"
    elif has("jean", "denim"):
        formality, movement, energy, aesthetic = 2, 6, 5, "casual"
    elif has("short"):
        formality, movement, energy, aesthetic = 1, 8, 6, "casual"
        avoid_for = list(_CASUAL_OCC)
    # ---- footwear casual ----
    elif has("sandal", "slide", "espadrille", "flip"):
        formality, movement, energy, aesthetic = 2, 6, 5, "resort"
        avoid_for = ["office", "wedding", "funeral", "interview"]
    elif has("boot", "chelsea"):
        formality, movement, energy, aesthetic = 3, 5, 5, "classic"
    # ---- accessories ----
    elif has("belt"):
        formality, movement, energy, aesthetic = 3, 5, 3, "classic"
        avoid_for = list(_FESTIVAL)
    elif has("cap", "baseball hat", "bucket hat"):
        formality, movement, energy, aesthetic = 1, 8, 6, "street"
        avoid_for = list(_CASUAL_OCC)
    elif has("sunglass", "eyewear"):
        formality, movement, energy, aesthetic = 2, 7, 6, "casual"
    elif has("bag", "tote", "crossbody", "clutch", "backpack"):
        formality, movement, energy, aesthetic = 2, 5, 4, "casual"
    elif has("watch", "ring", "necklace", "bracelet", "chain", "jewel"):
        formality, movement, energy, aesthetic = 3, 5, 5, "minimal"
    # ---- dresses (default mid) ----
    elif cat == "dress" or has("dress", "jumpsuit"):
        formality, movement, energy, aesthetic = 3, 5, 6, "feminine"

    return {
        "formality": formality,
        "movement": movement,
        "energy": energy,
        "aesthetic": aesthetic,
        "avoid_for": avoid_for,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--apply", action="store_true", help="write back (default: dry-run)")
    args = ap.parse_args()

    with open(args.file, encoding="utf-8") as f:
        raw_text = f.read()
    data = json.loads(raw_text)
    rows = data if isinstance(data, list) else next((v for v in data.values() if isinstance(v, list)), [])
    if not rows:
        print("No asset rows found.")
        return 1

    AXES = ("formality", "movement", "energy", "aesthetic", "avoid_for")
    changed = 0
    aesthetic_hist: Counter = Counter()
    sample = []
    for a in rows:
        if not isinstance(a, dict):
            continue
        derived = _derive(a)
        wrote = False
        for k in AXES:
            # additive: only fill if missing/empty
            if a.get(k) in (None, "", [], {}):
                a[k] = derived[k]
                wrote = True
        if wrote:
            changed += 1
        aesthetic_hist[a.get("aesthetic")] += 1
        if len(sample) < 8:
            sample.append((a.get("name"), a["formality"], a["movement"], a["energy"], a["aesthetic"], a.get("avoid_for")))

    print(f"file={args.file}  assets={len(rows)}  filled={changed}")
    print("aesthetic distribution:", dict(aesthetic_hist))
    print("--- sample ---")
    for s in sample:
        print(f"  {s[0][:34]:34s} f={s[1]} mv={s[2]} en={s[3]} {s[4]:12s} avoid={s[5]}")

    if args.apply:
        bak = args.file + ".bak"
        with open(bak, "w", encoding="utf-8") as f:
            f.write(raw_text)  # true pre-mutation original
        with open(args.file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"APPLIED -> {args.file}  (backup: {bak})")
    else:
        print("DRY-RUN. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
