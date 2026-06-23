"""DRY-RUN style_assets metadata backfill PROPOSER. Review-only.

Reads live style_assets, proposes fills for missing colors / occasions /
archetypes and a recommended action for invalid categories. Writes a proposal
file for human review. **No DB writes, no --apply path** — by design.

  python scripts/backfill_style_metadata_dryrun.py
  -> data/style_asset_backfill_proposal.csv  (asset_id, field, current, proposed, confidence, reason)
"""
from __future__ import annotations
import os, sys, csv, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("R2_LOAD_LOCAL_ENV", "true")
_envp = ROOT / ".env"
if _envp.exists():
    for line in _envp.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("="); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# reuse the fixed word-boundary colour inference
from scripts.patch_asset_colors import _infer_colors

ALLOWED_CAT = {"top", "bottom", "dress", "ethnic", "outerwear", "footwear", "accessory"}
VALID_GENDER = {"male", "female", "unisex"}


def lst(v):
    if isinstance(v, list): return [str(x).strip() for x in v if str(x).strip()]
    if v in (None, ""): return []
    return [str(v).strip()]


def infer_occasions(cat, name, sub):
    f = (name + " " + sub).lower(); o = set()
    if cat == "ethnic": o |= {"wedding", "festive"}
    if cat == "dress": o |= {"dinner", "party", "brunch"}
    if cat == "bottom":
        if "skirt" in f: o |= {"brunch", "coffee_date"}
        elif any(k in f for k in ["trouser", "pant", "formal"]): o |= {"office", "dinner"}
        elif "jean" in f: o |= {"dailywear", "coffee_date", "travel"}
        elif "short" in f: o |= {"dailywear", "travel"}
        else: o |= {"dailywear"}
    if cat == "top":
        o |= {"dailywear", "coffee_date"}
        if any(k in f for k in ["blouse", "shirt", "formal"]): o |= {"office", "dinner"}
    if cat == "outerwear":
        o |= {"office", "dinner"}
        if any(k in f for k in ["coat", "trench", "puffer"]): o |= {"travel"}
    if cat == "footwear":
        if any(k in f for k in ["heel", "formal", "loafer"]): o |= {"office", "party", "dinner"}
        elif "sneaker" in f: o |= {"dailywear", "travel"}
        else: o |= {"dailywear"}
    if cat == "accessory": o |= {"dailywear"}
    return sorted(o)


def infer_archetypes(cat, name, sub, cols):
    f = (name + " " + sub).lower()
    if cat == "ethnic": return ["Festive Heritage"]
    if "blazer" in f: return ["Modern Professional"]
    if cat == "bottom" and any(k in f for k in ["trouser", "formal"]) and any(c in cols for c in ["black", "navy", "charcoal", "grey", "beige"]):
        return ["Modern Professional"]
    if cat == "dress" and any(k in f for k in ["gown", "sequin", "satin", "velvet"]): return ["Modern Romantic"]
    if cat == "bottom" and "jean" in f: return ["Refined Weekend"]
    return []  # low confidence -> leave empty


def category_action(cat, name):
    f = name.lower()
    if cat == "grooming" or any(k in f for k in ["serum", "razor", "perfume", "deodorant", "toothpaste", "facewash", "lipbalm", "retinol"]):
        return ("deactivate", "non-fashion grooming/skincare; not a stylable asset", "high")
    if cat == "travel":
        if any(k in f for k in ["suitcase", "luggage", "passport", "pillow", "charger", "adapter", "bottle", "pouch"]):
            return ("deactivate", "non-apparel travel gadget", "high")
        return ("reclassify->accessory", "travel bag should be accessory(+subcategory)", "med")
    if cat == "loungewear":
        return ("keep_or_rename", "sleepwear; valid but non-canonical category — confirm taxonomy", "low")
    return ("review", "category not in canonical set", "low")


def main():
    from services.appwrite_proxy import AppwriteProxy
    p = AppwriteProxy()
    docs = []; off = 0
    while True:
        pg = p.list_documents("style_assets", limit=100, offset=off, return_meta=True)
        rows = [r for r in (pg.get("documents") or []) if isinstance(r, dict)]
        if not rows: break
        docs += rows; off += len(rows)
        if (pg.get("meta") or {}).get("has_more") == False: break
        if len(rows) < 100: break

    proposals = []
    stats = Counter()
    for d in docs:
        aid = str(d.get("asset_id") or d.get("$id") or "")
        name = str(d.get("name") or ""); sub = str(d.get("subcategory") or "")
        cat = str(d.get("category") or "").strip().lower()
        cols = lst(d.get("colors"))
        # colors
        if not cols:
            sug = _infer_colors(name, sub, aid)
            conf = "high" if sug else "low"
            proposals.append([aid, "colors", "[]", "|".join(sug) or "(none - needs image extraction)", conf,
                              "word-boundary infer from name/subcategory/asset_id"])
            stats["colors"] += 1
        # occasions
        if not lst(d.get("occasions")):
            sug = infer_occasions(cat, name, sub)
            proposals.append([aid, "occasions", "[]", "|".join(sug), "med" if sug else "low",
                              "category+keyword heuristic"])
            stats["occasions"] += 1
        # archetypes
        if not lst(d.get("archetypes")):
            sug = infer_archetypes(cat, name, sub, cols or _infer_colors(name, sub, aid))
            proposals.append([aid, "archetypes", "[]", "|".join(sug) or "(leave empty - low confidence)",
                              "high" if sug else "low", "category/occasion/color heuristic; empty when unsure"])
            stats["archetypes"] += 1
        # invalid category
        if cat not in ALLOWED_CAT:
            act, reason, conf = category_action(cat, name)
            proposals.append([aid, "category(%s)" % cat, cat, act, conf, reason])
            stats["category"] += 1
        # invalid gender
        if str(d.get("gender") or "").strip().lower() not in VALID_GENDER:
            proposals.append([aid, "gender", str(d.get("gender")), "(needs manual set)", "low", "gender missing/invalid"])
            stats["gender"] += 1

    out = ROOT / "data" / "style_asset_backfill_proposal.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["asset_id", "field", "current", "proposed", "confidence", "reason"])
        w.writerows(proposals)
    print("PROPOSAL (dry run, NO writes) ->", out)
    print("total assets scanned:", len(docs))
    print("proposed fixes:", dict(stats), "| total rows:", len(proposals))
    print("NOTE: review-only. No --apply path. colors -> apply via fixed scripts/patch_asset_colors.py after review.")


if __name__ == "__main__":
    main()
