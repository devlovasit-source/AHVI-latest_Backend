"""Board smoke harness — offline, no Gemini/DB.

Runs the deterministic board pipeline (occasion resolve -> brief -> archetypes ->
asset retrieval) over the seed catalog for 10 prompts and prints, per prompt:
board titles, selected items, missing images, bad items (forbidden signal or
formality > brief+2). Demo sanity check that music festival != oxford/belt/loafer.

Usage:
  STYLE_SHARED_BRAIN=true python scripts/board_smoke_harness.py
  python scripts/board_smoke_harness.py --file data/style_assets_mens_seed.json --gender male
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("STYLE_SHARED_BRAIN", "true")

from brain.engines import style_scorer as ss            # noqa: E402
from services import style_context_service as scs        # noqa: E402
from services import stylist_knowledge_service as sks    # noqa: E402
from services import style_reasoning_engine as e         # noqa: E402

PROMPTS = [
    "music festival", "coffee date", "airport travel", "wedding guest",
    "office meeting", "beach vacation", "sangeet", "brunch",
    "conference", "date night",
]


def _load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, list) else next((v for v in d.values() if isinstance(v, list)), [])


def _direction(arch):
    items = list(arch.get("preferred_items") or [])
    hero = items[0] if items else "shirt"
    return {
        "archetype": str(arch.get("name") or "").lower(),
        "title": arch.get("name"),
        "hero_piece": hero,
        "items": items[:4],
        "colors": list(arch.get("palette") or [])[:3],
    }


def _bad(name, brief):
    t = str(name or "").lower()
    forb = [s.lower() for s in (brief.get("forbidden_item_signals") or [])]
    if any(s in t for s in forb):
        return "forbidden_signal"
    bf = brief.get("formality")
    if isinstance(bf, (int, float)) and e._estimate_text_formality(t) > bf + 2:
        return f"too_formal(f{e._estimate_text_formality(t)}>brief{bf}+2)"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/style_assets_meghna_seed.json")
    ap.add_argument("--gender", default="female")
    args = ap.parse_args()

    assets = _load(args.file)
    missing_imgs = [a.get("name") for a in assets if isinstance(a, dict) and not (a.get("image_url") or a.get("imageUrl"))]

    print(f"catalog={args.file} assets={len(assets)} missing_image={len(missing_imgs)} gender={args.gender} flag={os.environ.get('STYLE_SHARED_BRAIN')}")
    print("=" * 100)

    for p in PROMPTS:
        brief = scs.build_canonical_style_context(query=p, router_occasion=p)
        retr = ss.normalize_occasion(p) or p
        archs = sks.select_archetypes(occasion=brief["canonical_occasion"], gender=args.gender, limit=4)
        print(f"\nPROMPT: {p}")
        print(f"  canonical={brief['canonical_occasion']}  family={brief['occasion_family']}  "
              f"formality={brief.get('formality')} energy={brief.get('energy')} movement={brief.get('movement')}  retr_occ={retr}")
        if brief.get("forbidden_archetypes"):
            print(f"  forbidden_arch={brief['forbidden_archetypes'][:3]}... item_signals={len(brief.get('forbidden_item_signals') or [])}")
        for arch in archs[:2]:
            d = _direction(arch)
            hero = e._best_style_asset(assets, direction=d, occasion=retr, target_gender=args.gender, brief=brief)
            acc = e._best_style_assets(assets, direction=d, occasion=retr, accessory_only=True,
                                       target_gender=args.gender, limit=3, brief=brief)
            hero_name = hero.get("name") if hero else "-(none scored>0)"
            acc_names = [a.get("name") for a in acc]
            picks = ([hero_name] if hero else []) + acc_names
            bad = [(n, _bad(n, brief)) for n in picks if _bad(n, brief)]
            print(f"  BOARD: {arch.get('name')}")
            print(f"    hero: {hero_name}")
            print(f"    complete_the_look: {acc_names}")
            if bad:
                print(f"    !! BAD ITEMS: {bad}")
    print("\n" + "=" * 100)
    if missing_imgs:
        print(f"MISSING IMAGES ({len(missing_imgs)}): {missing_imgs[:10]}")
    else:
        print("MISSING IMAGES: none in catalog")
    return 0


if __name__ == "__main__":
    sys.exit(main())
