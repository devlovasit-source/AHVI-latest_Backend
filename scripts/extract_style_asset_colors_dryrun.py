"""DRY-RUN image dominant-colour extractor for style_assets.

Per P1_COLOR_EXTRACTOR_DESIGN.md. Reads live style_assets (read-only), fetches
each image_url over HTTPS (GET; r2.dev blocks HEAD), extracts 1-3 canonical
colours via Pillow median-cut + HSV mapping, cross-checks filename inference,
and writes a PROPOSAL CSV. **No DB writes, no R2 writes, no image mutation.**

    python scripts/extract_style_asset_colors_dryrun.py            # all assets
    python scripts/extract_style_asset_colors_dryrun.py --limit 20 # sample
"""
from __future__ import annotations
import os, sys, io, csv, time, argparse, colorsys, urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("R2_LOAD_LOCAL_ENV", "true")
_envp = ROOT / ".env"
if _envp.exists():
    for line in _envp.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("="); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from PIL import Image
from scripts.patch_asset_colors import _infer_colors  # fixed word-boundary text infer

OUT = ROOT / "data" / "style_asset_color_image_proposal.csv"
ALLOWED_HOSTS = ("pub-43484c7ec0d741cabcac4df01e98344b.r2.dev", "image.hm.com")
LOWCONF_CATS = {"accessory", "jewellery", "jewelry", "grooming"}
TIMEOUT = 15
MAX_BYTES = 6 * 1024 * 1024
RATE_LIMIT = 0.15
THUMB = (128, 128)
MIN_SHARE = 0.12
HIGH_SHARE = 0.55


# ---------- canonical colour mapping (RGB -> name) ----------
def rgb_to_name(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    h *= 360
    if v < 0.16:
        return "black"
    if s < 0.12:
        if v > 0.85:
            return "white"
        if v > 0.62:
            return "silver"
        return "grey"
    if s < 0.32 and v > 0.78 and 25 <= h <= 70:
        return "cream"
    if s <= 0.5 and 0.5 <= v <= 0.82 and 20 <= h <= 55:
        return "beige"
    if h < 15 or h >= 345:
        if v < 0.5:
            return "burgundy" if h >= 330 else "maroon"
        return "red"
    if 15 <= h < 45:
        if v < 0.55 and s > 0.4:
            return "brown"
        if s < 0.6 and v > 0.7:
            return "gold"
        return "orange"
    if 45 <= h < 66:
        return "yellow"
    if 66 <= h < 90 and v < 0.55:
        return "olive"
    if 66 <= h < 160:
        return "green"
    if 160 <= h < 250:
        return "navy" if v < 0.4 else "blue"
    if 250 <= h < 295:
        return "purple"
    return "pink"  # 295-345


def _is_bg(r, g, b, corner):
    # near-white / near-black background, or close to sampled corner colour
    if min(r, g, b) > 238:
        return True
    if max(r, g, b) < 16:
        return True
    if corner and (abs(r - corner[0]) + abs(g - corner[1]) + abs(b - corner[2])) < 36:
        return True
    return False


def extract_colors(img: Image.Image):
    """Returns (names[1..3], dominant_hex, secondary_hexes, garment_share, n_clusters)."""
    img = img.copy()
    img.thumbnail(THUMB)
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    rgba = img.convert("RGBA")
    px = list(rgba.getdata())
    w, h = rgba.size
    # corner background sample (median of 4 corners)
    corners = [px[0], px[w - 1], px[(h - 1) * w], px[h * w - 1]]
    corner = tuple(sorted(c[i] for c in corners)[1] for i in range(3))
    fg = []
    for (r, g, b, a) in px:
        if has_alpha and a < 16:
            continue
        if _is_bg(r, g, b, corner):
            continue
        fg.append((r, g, b))
    total = len([1 for p in px if not (has_alpha and p[3] < 16)])
    garment_share = (len(fg) / total) if total else 0.0
    white_black_fallback = False
    if len(fg) < 0.12 * max(1, total):
        # garment likely IS the background tone (white/black on same bg)
        fg = [(r, g, b) for (r, g, b, a) in px if not (has_alpha and a < 16)]
        white_black_fallback = True
    if not fg:
        return [], "", "", 0.0, 0
    # median-cut quantize via a temp image of fg pixels
    side = max(1, int(len(fg) ** 0.5))
    tmp = Image.new("RGB", (side, side))
    tmp.putdata((fg + fg)[: side * side])
    q = tmp.quantize(colors=6, method=Image.MEDIANCUT)
    pal = q.getpalette()
    counts = Counter(q.getdata())
    clusters = []
    for idx, cnt in counts.most_common():
        rgb = (pal[idx * 3], pal[idx * 3 + 1], pal[idx * 3 + 2])
        clusters.append((rgb, cnt))
    npx = sum(c for _, c in clusters) or 1
    # Accumulate share PER canonical name (merge shade-buckets of same colour),
    # keep the first/biggest hex per name.
    name_share, name_hex = {}, {}
    for rgb, cnt in clusters:
        nm = rgb_to_name(*rgb)
        name_share[nm] = name_share.get(nm, 0) + cnt
        if nm not in name_hex:
            name_hex[nm] = "#%02x%02x%02x" % rgb
    ranked = sorted(name_share.items(), key=lambda kv: -kv[1])
    names, hexes = [], []
    for nm, cnt in ranked:
        if cnt / npx < MIN_SHARE and names:
            break
        names.append(nm); hexes.append(name_hex[nm])
        if len(names) >= 3:
            break
    top_share = ranked[0][1] / npx if ranked else 0.0
    dom_hex = hexes[0] if hexes else ""
    sec = "|".join(hexes[1:]) if len(hexes) > 1 else ""
    if white_black_fallback and names:
        names = [names[0]]; hexes = hexes[:1]; sec = ""
        top_share = 0.4
    return names, dom_hex, sec, top_share, len(ranked)


def fetch(url: str) -> bytes:
    if not url.lower().startswith("https://") or not any(host in url for host in ALLOWED_HOSTS):
        raise ValueError("unsafe/unknown host")
    req = urllib.request.Request(url, headers={"User-Agent": "ahvi-color-extractor/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        ctype = r.headers.get("content-type", "")
        if not ctype.startswith("image/"):
            raise ValueError("non-image content-type: %s" % ctype)
        return r.read(MAX_BYTES + 1)[:MAX_BYTES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

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
    if args.limit:
        docs = docs[: args.limit]

    def lst(v):
        if isinstance(v, list): return [str(x).strip() for x in v if str(x).strip()]
        return [str(v).strip()] if v not in (None, "") else []

    rows_out = []
    conf_counter = Counter()
    missing_before = sum(1 for d in docs if not lst(d.get("colors")))
    for d in docs:
        aid = str(d.get("asset_id") or d.get("$id") or "")
        name = str(d.get("name") or ""); cat = str(d.get("category") or "").lower()
        url = str(d.get("image_url") or "")
        cur = lst(d.get("colors"))
        rec = dict(asset_id=aid, name=name, category=cat, image_url=url,
                   current_colors="|".join(cur), suggested_colors="", dominant_hex="",
                   secondary_hexes="", confidence="", method="", reason="", status="")
        if cur:
            rec.update(method="existing", status="has_colors", confidence="n/a",
                       suggested_colors="|".join(cur), reason="already has colors")
            rows_out.append(rec); continue
        try:
            data = fetch(url)
            img = Image.open(io.BytesIO(data))
            names, dom, sec, share, ncl = extract_colors(img)
            time.sleep(RATE_LIMIT)
            if not names:
                rec.update(method="image_quantize", status="low_conf", confidence="low",
                           reason="no garment pixels / cluster fail")
                conf_counter["low"] += 1; rows_out.append(rec); continue
            fn_colors = set(_infer_colors(name, d.get("subcategory"), aid))
            agree = bool(fn_colors & set(names))
            conflict = bool(fn_colors) and not agree
            if cat in LOWCONF_CATS:
                conf = "low"; reason = "accessory/jewellery/grooming — unreliable"
            elif conflict:
                conf = "medium"; reason = "text/image colour conflict (text=%s)" % "|".join(sorted(fn_colors))
            elif len(names) == 1 and share >= HIGH_SHARE:
                conf = "high"; reason = "single dominant garment colour%s" % (" (agrees w/ name)" if agree else "")
            else:
                conf = "medium"; reason = "%d colours / share=%.2f" % (len(names), share)
            rec.update(suggested_colors="|".join(names), dominant_hex=dom, secondary_hexes=sec,
                       confidence=conf, method="image_quantize", reason=reason, status="ok")
            conf_counter[conf] += 1
        except Exception as e:
            rec.update(method="image_quantize", status="failed", confidence="low",
                       reason=str(e)[:120])
            conf_counter["failed"] += 1
        rows_out.append(rec)

    cols = ["asset_id", "name", "category", "image_url", "current_colors", "suggested_colors",
            "dominant_hex", "secondary_hexes", "confidence", "method", "reason", "status"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows_out: w.writerow(r)

    proposed = sum(1 for r in rows_out if r["status"] == "ok")
    print("PROPOSAL ->", OUT)
    print("assets scanned:", len(docs))
    print("missing colors before:", missing_before)
    print("proposals (status=ok):", proposed)
    print("confidence:", dict(conf_counter))
    print("\nsample ok rows:")
    for r in [x for x in rows_out if x["status"] == "ok"][:6]:
        print("  %-44s %-18s %s %s" % (r["asset_id"][:44], r["suggested_colors"], r["confidence"], r["dominant_hex"]))


if __name__ == "__main__":
    main()
