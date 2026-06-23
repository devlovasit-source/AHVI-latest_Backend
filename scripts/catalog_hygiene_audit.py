"""DRY-RUN catalog hygiene audit: flag non-fashion contamination in style_assets.

Classifies every asset as fashion (KEEP) or non-fashion (FLAG) from category +
name. Read-only. Writes data/style_asset_hygiene_proposal.csv. No DB writes.

    python scripts/catalog_hygiene_audit.py
"""
from __future__ import annotations
import os, sys, csv
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

OUT = ROOT / "data" / "style_asset_hygiene_proposal.csv"

FASHION_CATS = {"top", "bottom", "dress", "ethnic", "outerwear", "footwear", "accessory"}

# Non-fashion keyword -> predicted_type (checked FIRST; wins over category).
NONFASHION = [
    (("weighing", "weight scale", "weighing scale"), "appliance"),
    (("mouth freshener", "freshener", "toothpaste", "tooth brush", "toothbrush", "sanitizer", "razor", "tweezer", "nail clipper"), "hygiene"),
    (("document holder", "documentholder", "file organiser", "file organizer", "fileorganiser", "passport", "luggage tag", "organiser", "organizer", "storage box", "storagebox"), "travel_utility"),
    (("medical kit", "medicalkit", "first aid", "medicine", "pill"), "medical"),
    (("charger", "powerbank", "power bank", "adapter", "cable", "earphone", "earbud", "headphone", "electronic", "weighingscale", "handfan", "hand fan", "torch"), "electronics_gadget"),
    (("neck pillow", "pillow", "umbrella", "water bottle", "bottle", "pouch only", "travelpouch"), "travel_gadget"),
    (("serum", "lotion", "moistur", "cleanser", "facewash", "face wash", "sunscreen", "toner", "lipstick", "kajal", "mascara", "foundation", "makeup", "perfume", "deodorant", "skincare", "retinol", "lip balm", "lipbalm"), "beauty_grooming"),
    (("sports bra", "bralette", "pantyliner", "panty", "brief", "lingerie", "shapewear", "stick on bra", "strapless bra"), "innerwear"),
]
# fashion-positive keywords for confidence boost
FASHION_KW = ("shirt", "top", "tee", "t-shirt", "blouse", "dress", "gown", "skirt", "jean",
    "trouser", "pant", "short", "kurta", "kurti", "saree", "lehenga", "anarkali", "sharara",
    "blazer", "jacket", "coat", "cardigan", "shrug", "dupatta", "heel", "flat", "sneaker",
    "sandal", "loafer", "boot", "jutti", "mule", "wedge", "bag", "tote", "clutch", "handbag",
    "wallet", "earring", "necklace", "bracelet", "watch", "scarf", "belt", "sunglass", "tikka",
    "kurta set", "co-ord", "jumpsuit", "camisole", "corset", "polo", "sweater", "knit")


def classify(cat, name):
    blob = (name + " " + cat).lower()
    for kws, ptype in NONFASHION:
        if any(k in blob for k in kws):
            return ptype, "no", "high", "non-fashion keyword (%s)" % ptype
    # fashion by keyword
    if any(k in blob for k in FASHION_KW):
        return ("apparel/fashion", "yes", "high", "fashion keyword + category=%s" % cat)
    # fashion by clean category, no keyword
    if cat in FASHION_CATS:
        return ("apparel/fashion", "yes", "medium", "fashion category=%s, no garment keyword" % cat)
    # unknown / legacy categories (travel, loungewear, grooming)
    if cat in ("grooming",):
        return ("beauty_grooming", "no", "high", "category=grooming")
    if cat in ("travel",):
        return ("travel_utility", "review", "low", "category=travel (mixed bag/gadget) — manual review")
    if cat in ("loungewear",):
        return ("loungewear", "review", "low", "sleepwear — taxonomy/keep decision")
    return ("unknown", "review", "low", "no signal; manual review")


def main():
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

    out = []
    for d in docs:
        aid = str(d.get("asset_id") or d.get("$id") or "")
        name = str(d.get("name") or ""); cat = str(d.get("category") or "").lower()
        ptype, is_fashion, conf, reason = classify(cat, name)
        out.append(dict(asset_id=aid, asset_name=name, category=cat,
                        image_url=str(d.get("image_url") or ""), predicted_type=ptype,
                        is_fashion_asset=is_fashion, confidence=conf, reason=reason))

    # FLAG (non-fashion) first for review convenience
    order = {"no": 0, "review": 1, "yes": 2}
    out.sort(key=lambda r: (order.get(r["is_fashion_asset"], 3), r["confidence"]))
    cols = ["asset_id", "asset_name", "category", "image_url", "predicted_type",
            "is_fashion_asset", "confidence", "reason"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in out: w.writerow(r)

    flag = [r for r in out if r["is_fashion_asset"] == "no"]
    review = [r for r in out if r["is_fashion_asset"] == "review"]
    print("HYGIENE PROPOSAL ->", OUT)
    print("total:", len(out), "| KEEP(fashion):", sum(1 for r in out if r["is_fashion_asset"] == "yes"),
          "| FLAG(non-fashion):", len(flag), "| REVIEW:", len(review))
    print("flagged predicted_type:", dict(Counter(r["predicted_type"] for r in flag)))
    print("\nFLAGGED non-fashion (deactivate candidates):")
    for r in flag[:30]:
        print("  %-46s %-18s %s" % (r["asset_id"][:46], r["predicted_type"], r["asset_name"][:30]))


if __name__ == "__main__":
    main()
