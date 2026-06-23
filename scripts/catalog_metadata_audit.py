"""Catalog metadata audit — READ ONLY. No edits, no deploy, no DB writes.

Audits the seed catalogs that feed style boards. Defaults to the *.bak
snapshots = pre-backfill = representative of the LIVE Appwrite catalog (the
Task-2 axes are local + un-imported). Pass --live to audit the working-tree
seeds instead (post-backfill, to confirm the backfill).

Reports: field coverage %, category/subcategory/occasion/archetype distros,
duplicates, naming inconsistency, festival-vs-formal mis-tags, top-N to fix.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict

STYLE_AXES = ("formality", "movement", "energy", "aesthetic", "avoid_for")
CORE = ("asset_id", "name", "category", "subcategory", "color", "colors", "occasions", "archetypes", "style_tags", "image_url")

FORMAL_WORDS = ("oxford", "loafer", "belt", "blazer", "suit", "derby", "brogue", "trouser", "dress shoe", "tie")
CASUAL_WORDS = ("tee", "t-shirt", "graphic", "sneaker", "cargo", "hoodie", "jogger", "short", "cap", "tank", "slides")


def _load(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        r = d if isinstance(d, list) else next((v for v in d.values() if isinstance(v, list)), [])
        for a in r:
            if isinstance(a, dict):
                a["_src"] = p.split("/")[-1].split("\\")[-1]
                rows.append(a)
    return rows


def _norm_name(n):
    n = str(n or "").lower()
    n = re.sub(r"\b(h&m|h and m|mens|men|womens|women|the)\b", " ", n)
    n = re.sub(r"\b\d+\b", " ", n)           # drop variant numbers
    n = re.sub(r"[^a-z ]", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def pct(c, n):
    return f"{(c*100//n) if n else 0}% ({c}/{n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="audit working-tree seeds (post-backfill)")
    args = ap.parse_args()
    pat = "data/style_assets_*seed.json" if args.live else "data/style_assets_*seed.json.bak"
    paths = [p for p in glob.glob(pat) if "risky" not in p and "manifest" not in p and "result" not in p and "plan" not in p]
    rows = _load(paths)
    n = len(rows)
    print(f"SOURCE: {'LIVE working-tree' if args.live else 'PRE-BACKFILL .bak (live-representative)'}")
    print(f"files: {[p.split(chr(92))[-1].split('/')[-1] for p in paths]}")
    print(f"TOTAL ASSETS: {n}")
    print("=" * 90)

    # 1. field coverage
    print("\n[1] FIELD COVERAGE (non-empty)")
    for f in CORE + STYLE_AXES:
        c = sum(1 for a in rows if a.get(f) not in (None, "", [], {}))
        print(f"  {f:14s} {pct(c, n)}")

    # 2. category distribution (consistency)
    print("\n[2] CATEGORY VALUES (consistency)")
    print("  ", dict(Counter(str(a.get('category') or 'NONE') for a in rows)))
    print("[2b] SUBCATEGORY top-20")
    sub = Counter(str(a.get('subcategory') or 'NONE') for a in rows)
    for k, v in sub.most_common(20):
        print(f"    {k:24s} {v}")

    # 3. occasion coverage
    print("\n[3] OCCASION COVERAGE")
    empty = sum(1 for a in rows if not a.get("occasions"))
    daily_only = sum(1 for a in rows if [str(x).lower() for x in (a.get("occasions") or [])] in (["daily"], ["dailywear"]))
    print(f"  occasions empty:      {pct(empty, n)}")
    print(f"  occasions daily-only: {pct(daily_only, n)}")
    occ = Counter()
    for a in rows:
        for o in (a.get("occasions") or []):
            occ[str(o).lower()] += 1
    print("  occasion vocab:", dict(occ.most_common()))

    # 4. archetype coverage
    print("\n[4] ARCHETYPE COVERAGE")
    ac = sum(1 for a in rows if a.get("archetypes"))
    print(f"  archetypes non-empty: {pct(ac, n)}")
    arch = Counter()
    for a in rows:
        for x in (a.get("archetypes") or []):
            arch[str(x).lower()] += 1
    print("  archetype vocab:", dict(arch.most_common(15)) or "—EMPTY—")

    # 5. style metadata coverage
    print("\n[5] STYLE METADATA COVERAGE")
    for f in STYLE_AXES:
        c = sum(1 for a in rows if a.get(f) not in (None, "", [], {}))
        print(f"  {f:11s} {pct(c, n)}")

    # duplicates by normalized name
    print("\n[6] DUPLICATE GARMENTS (same normalized name, 2+)")
    byname = defaultdict(list)
    for a in rows:
        byname[_norm_name(a.get("name"))].append(a.get("name"))
    dups = {k: v for k, v in byname.items() if len(v) > 1 and k}
    print(f"  duplicate name-clusters: {len(dups)}  (assets involved: {sum(len(v) for v in dups.values())})")
    for k, v in sorted(dups.items(), key=lambda x: -len(x[1]))[:12]:
        print(f"    '{k}': {v[:5]}{'...' if len(v) > 5 else ''} (x{len(v)})")

    # naming inconsistency
    print("\n[7] NAMING ISSUES")
    no_space = [a.get("name") for a in rows if a.get("name") and " " not in str(a.get("name"))]
    brand = [a.get("name") for a in rows if re.match(r"(?i)^(h&m|mens|womens)", str(a.get("name") or ""))]
    numbered = [a.get("name") for a in rows if re.search(r"\b\d+\b", str(a.get("name") or ""))]
    print(f"  no-space names (glued): {len(no_space)}  e.g. {no_space[:5]}")
    print(f"  brand-prefixed:         {len(brand)}  e.g. {brand[:3]}")
    print(f"  numbered variants:      {len(numbered)}  e.g. {numbered[:3]}")

    # festival vs formal mis-tag
    print("\n[8] FESTIVAL-vs-FORMAL MIS-TAG")
    formal_named = [a for a in rows if any(w in str(a.get("name") or "").lower() for w in FORMAL_WORDS)]
    festival_formal = [a.get("name") for a in formal_named
                       if any(o in ("party", "festival", "rave", "concert") for o in [str(x).lower() for x in (a.get("occasions") or [])])]
    print(f"  formal-named assets (oxford/loafer/belt/blazer/...): {len(formal_named)}")
    print(f"  of those tagged party/festival/rave/concert: {len(festival_formal)}  e.g. {festival_formal[:5]}")
    office_casual = [a.get("name") for a in rows
                     if any(w in str(a.get("name") or "").lower() for w in CASUAL_WORDS)
                     and "office" in [str(x).lower() for x in (a.get("occasions") or [])]]
    print(f"  casual-named assets tagged office: {len(office_casual)}  e.g. {office_casual[:5]}")

    # top-N needing correction
    print("\n[9] TOP 50 ASSETS NEEDING CORRECTION (heuristic)")
    scored = []
    for a in rows:
        miss = 0
        if not a.get("subcategory") or a.get("subcategory") in (None, "", "other"): miss += 2
        if not a.get("occasions"): miss += 3
        if [str(x).lower() for x in (a.get("occasions") or [])] in (["daily"], ["dailywear"]): miss += 2
        if not a.get("archetypes"): miss += 1
        if not (a.get("color") or a.get("colors")): miss += 1
        if " " not in str(a.get("name") or ""): miss += 1
        for f in STYLE_AXES:
            if a.get(f) in (None, "", [], {}): miss += 1
        if miss:
            scored.append((miss, a.get("asset_id"), a.get("name"), a.get("category"), a.get("subcategory"), a.get("occasions"), a.get("archetypes")))
    scored.sort(key=lambda x: -x[0])
    print(f"  assets with >=1 gap: {len(scored)}/{n}")
    for s in scored[:50]:
        print(f"    fixscore={s[0]:2d} | {str(s[2])[:30]:30s} | cat={s[3]} sub={s[4]} occ={s[5]} arch={s[6]}")


if __name__ == "__main__":
    main()
