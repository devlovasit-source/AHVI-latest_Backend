
import argparse
import json
import os
import re
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("APPWRITE_PROXY_LOAD_LOCAL_ENV", "true")
os.environ.setdefault("R2_LOAD_LOCAL_ENV", "true")

import requests
from PIL import Image

from services.appwrite_proxy import AppwriteProxy
from services.bg_service import remove_bg_external_sync
from services.r2_storage import R2Storage


def text(v: Any) -> str:
    return str(v or "").strip()


def slug(v: str) -> str:
    v = re.sub(r"[^a-zA-Z0-9]+", "_", text(v).lower()).strip("_")
    return v or "asset"


def role_of(asset: Dict[str, Any]) -> str:
    raw = text(asset.get("role") or asset.get("category") or asset.get("item_type") or asset.get("type")).lower()
    name = text(asset.get("name") or asset.get("title")).lower()

    if raw in {"tops", "top", "shirt", "tshirt", "t-shirt", "blouse"}:
        return "top"
    if raw in {"bottoms", "bottom", "pants", "trousers", "jeans", "skirt", "shorts"}:
        return "bottom"
    if raw in {"footwear", "shoes", "shoe", "sneakers", "sandals", "heels", "boots"}:
        return "footwear"
    if raw in {"dresses", "dress", "ethnic", "saree", "sari", "lehenga"}:
        return "dress"
    if raw in {"outerwear", "jacket", "coat", "blazer"}:
        return "outerwear"
    if raw in {"accessories", "accessory", "bag", "watch", "belt", "jewelry", "jewellery"}:
        return "accessory"

    if any(x in name for x in ["saree", "sari", "lehenga", "dress"]):
        return "dress"
    if any(x in name for x in ["shoe", "sneaker", "sandal", "heel", "boot", "loafer", "flip flop"]):
        return "footwear"
    if any(x in name for x in ["jacket", "coat", "blazer"]):
        return "outerwear"
    if any(x in name for x in ["jean", "trouser", "pant", "skirt", "short"]):
        return "bottom"
    if any(x in name for x in ["shirt", "tshirt", "t-shirt", "top", "blouse", "tank"]):
        return "top"

    return raw or "asset"


def source_url(asset: Dict[str, Any]) -> str:
    for key in [
        "catalog_image_url",
        "image_url",
        "imageUrl",
        "url",
        "r2_url",
        "r2Url",
        "raw_image_url",
        "rawImageUrl",
        "original_image_url",
        "transparent_image_url",
        "cutout_url",
    ]:
        val = text(asset.get(key))
        if val.startswith("http"):
            return val
    return ""


def target_key(asset: Dict[str, Any]) -> str:
    existing = text(asset.get("r2_key") or asset.get("r2Key") or asset.get("board_r2_key") or asset.get("boardR2Key"))
    gender = slug(asset.get("gender") or "unknown")
    role = role_of(asset)
    name = slug(asset.get("name") or asset.get("title") or asset.get("$id") or "asset")

    if existing:
        base = existing.rsplit(".", 1)[0].replace(" ", "_")
        if base.endswith("_cutout"):
            return base + ".png"
        return base + "_cutout.png"

    return f"style-assets/all/{gender}/{role}/{name}_cutout.png"


def has_png_board(asset: Dict[str, Any]) -> bool:
    board = text(asset.get("board_image_url") or asset.get("boardImageUrl"))
    return board.lower().endswith(".png")


def list_assets(limit: int) -> List[Dict[str, Any]]:
    proxy = AppwriteProxy()
    rows = []
    offset = 0

    while len(rows) < limit:
        page = proxy.list_documents("style_assets", limit=min(100, limit - len(rows)), offset=offset, return_meta=True)
        docs = page.get("documents") if isinstance(page, dict) else page
        docs = [d for d in (docs or []) if isinstance(d, dict)]
        rows.extend(docs)
        if len(docs) < 100:
            break
        offset += 100

    return rows


def download(url: str) -> bytes:
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return r.content


def normalize_png_with_alpha(image_bytes: bytes) -> bytes:
    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    alpha = img.getchannel("A")
    sample = list(alpha.resize((96, 96)).getdata())
    transparent = sum(1 for v in sample if v < 250)

    if transparent < 20:
        raise RuntimeError(f"no_transparent_pixels transparent_sample={transparent}")

    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def upload_png(storage: R2Storage, *, key: str, image_bytes: bytes) -> str:
    bucket = storage.style_boards_bucket or storage.raw_bucket
    public_base = storage.style_boards_public_url or storage.raw_public_url

    if not bucket or not public_base:
        raise RuntimeError("Missing style-boards R2 bucket/public URL configuration.")

    storage._client().put_object(
        bucket,
        key,
        BytesIO(image_bytes),
        length=len(image_bytes),
        content_type="image/png",
    )

    return public_base.rstrip("/") + "/" + key.lstrip("/")


def main():
    parser = argparse.ArgumentParser(description="Generate transparent board PNGs for all eligible style assets.")
    parser.add_argument("--apply", action="store_true", help="Write R2/Appwrite updates. Default is dry-run.")
    parser.add_argument("--scan-limit", type=int, default=3000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--category", default="", help="Optional role filter: tops, bottoms, footwear, dresses, outerwear, accessories.")
    parser.add_argument("--force", action="store_true", help="Reprocess even assets that already have PNG board_image_url.")
    args = parser.parse_args()

    if not os.getenv("RMBG_SERVICE_URL"):
        raise SystemExit("RMBG_SERVICE_URL is missing. Stop.")

    proxy = AppwriteProxy()
    storage = R2Storage()

    rows = list_assets(args.scan_limit)

    wanted = []
    cat = args.category.lower().strip().rstrip("s")

    for asset in rows:
        doc_id = text(asset.get("$id"))
        if not doc_id:
            continue

        role = role_of(asset)

        if cat and role != cat:
            continue

        src = source_url(asset)
        if not src:
            continue

        if not args.force and has_png_board(asset):
            continue

        wanted.append(asset)

    if args.limit:
        wanted = wanted[: args.limit]

    print(json.dumps({
        "event": "AHVI_STYLE_ASSET_ALL_CUTOUT_SELECTED",
        "dry_run": not args.apply,
        "count": len(wanted),
        "by_role": dict(Counter(role_of(a) for a in wanted)),
        "by_gender": dict(Counter(text(a.get("gender") or "unknown") for a in wanted)),
        "samples": [
            {
                "asset_id": a.get("$id"),
                "gender": text(a.get("gender") or "unknown"),
                "name": text(a.get("name") or a.get("title")),
                "role": role_of(a),
                "target_r2_key": target_key(a),
            }
            for a in wanted[:10]
        ],
    }, ensure_ascii=False))

    stats = Counter()

    for asset in wanted:
        doc_id = text(asset.get("$id"))
        role = role_of(asset)
        key = target_key(asset)

        try:
            raw = download(source_url(asset))
            cutout = remove_bg_external_sync(raw)
            png = normalize_png_with_alpha(cutout)

            if args.apply:
                public_url = upload_png(storage, key=key, image_bytes=png)
                proxy.update_document(
                    "style_assets",
                    doc_id,
                    {
                        "board_image_url": public_url,
                        "board_r2_key": key,
                        "cutout_status": "ready",
                        "catalog_image_url": source_url(asset),
                    },
                )
            else:
                public_url = "dry-run://" + key

            stats["processed"] += 1
            stats["ready"] += 1
            stats[f"role:{role}"] += 1

            print(json.dumps({
                "event": "AHVI_STYLE_ASSET_ALL_CUTOUT_READY",
                "asset_id": doc_id,
                "role": role,
                "url": public_url,
            }, ensure_ascii=False))

        except Exception as e:
            stats["processed"] += 1
            stats["failed"] += 1
            print(json.dumps({
                "event": "AHVI_STYLE_ASSET_ALL_CUTOUT_FAILED",
                "asset_id": doc_id,
                "role": role,
                "error": str(e),
            }, ensure_ascii=False))

            if args.apply:
                try:
                    proxy.update_document("style_assets", doc_id, {"cutout_status": "failed"})
                except Exception:
                    pass

    by_role = {}
    for k, v in stats.items():
        if k.startswith("role:"):
            by_role[k.split(":", 1)[1]] = v

    result = {
        "success": stats["failed"] == 0,
        "dry_run": not args.apply,
        "stats": {
            "processed": stats["processed"],
            "ready": stats["ready"],
            "failed": stats["failed"],
            "by_role": by_role,
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
