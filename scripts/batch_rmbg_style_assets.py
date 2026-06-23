from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
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


P0_TARGETS = {
    "top": 20,
    "bottom": 15,
    "footwear": 15,
    "dress": 10,
    "outerwear": 8,
    "accessory": 10,
}
ROLE_PRIORITY = ("top", "dress", "bottom", "footwear", "outerwear", "accessory")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _text(value).lower()).strip()


def _list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_text(v) for v in value if _text(v)]
    text = _text(value)
    if not text:
        return []
    return [p.strip() for p in re.split(r"[,|]", text) if p.strip()]


def _asset_role(asset: Dict[str, Any]) -> str:
    blob = _norm(
        " ".join(
            [
                _text(asset.get("name")),
                _text(asset.get("category")),
                _text(asset.get("subcategory")),
                " ".join(_list(asset.get("tags"))),
            ]
        )
    )
    tokens = set(blob.split())
    if tokens & {"dress", "gown", "saree", "sari", "lehenga", "anarkali", "jumpsuit"}:
        return "dress"
    if tokens & {"blazer", "jacket", "coat", "outerwear", "overshirt", "cardigan"}:
        return "outerwear"
    if tokens & {"shoe", "shoes", "footwear", "loafer", "loafers", "sneaker", "sneakers", "sandal", "sandals", "boot", "boots", "heel", "heels", "juti", "jutti", "mojari"}:
        return "footwear"
    if tokens & {"trouser", "trousers", "pant", "pants", "jean", "jeans", "denim", "chino", "chinos", "skirt", "shorts", "bottom", "bottoms"}:
        return "bottom"
    if tokens & {"shirt", "tee", "tshirt", "t", "polo", "top", "blouse", "kurta", "sweater", "knit"}:
        return "top"
    if tokens & {"watch", "belt", "bag", "tote", "clutch", "cap", "hat", "sunglasses", "necklace", "earrings", "bracelet", "ring", "accessory", "accessories"}:
        return "accessory"
    return ""


def _doc_id(asset: Dict[str, Any]) -> str:
    return _text(asset.get("$id") or asset.get("asset_id") or asset.get("assetId") or asset.get("id"))


def _source_url(asset: Dict[str, Any]) -> str:
    return _text(asset.get("image_url") or asset.get("imageUrl") or asset.get("catalog_image_url") or asset.get("catalogImageUrl"))


def _target_key(asset: Dict[str, Any]) -> str:
    existing = _text(asset.get("r2_key") or asset.get("r2Key") or asset.get("board_r2_key") or asset.get("boardR2Key"))
    if not existing:
        parsed = urlparse(_source_url(asset))
        existing = parsed.path.lstrip("/")
    if not existing:
        safe_name = re.sub(r"[^a-z0-9._-]+", "_", _norm(asset.get("name"))) or _doc_id(asset) or "style_asset"
        existing = f"style-assets/{safe_name}.jpg"
    path = Path(existing)
    stem = path.stem
    parent = str(path.parent).replace("\\", "/")
    filename = f"{stem}_cutout.png"
    return filename if parent in {"", "."} else f"{parent}/{filename}"


def _download(url: str) -> bytes:
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    return response.content


def validate_alpha_png(image_bytes: bytes) -> Tuple[bool, str]:
    try:
        img = Image.open(BytesIO(image_bytes))
        if img.format != "PNG":
            return False, "not_png"
        if img.mode not in {"RGBA", "LA", "P"}:
            return False, "missing_alpha_mode"
        rgba = img.convert("RGBA")
        alpha = rgba.getchannel("A")
        values = list(alpha.resize((96, 96)).getdata())
        transparent = sum(1 for value in values if value < 16)
        opaque = sum(1 for value in values if value > 240)
        if transparent <= 0:
            return False, "no_transparent_pixels"
        if opaque <= 0:
            return False, "no_opaque_pixels"
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"invalid_image:{str(exc)[:80]}"


def _encode_png(image_bytes: bytes) -> bytes:
    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _upload_png(storage: R2Storage, *, key: str, image_bytes: bytes) -> str:
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
    return f"{public_base.rstrip('/')}/{key}"


def _list_style_assets(limit: int) -> List[Dict[str, Any]]:
    proxy = AppwriteProxy()
    rows: List[Dict[str, Any]] = []
    offset = 0
    while len(rows) < limit:
        page = proxy.list_documents("style_assets", limit=min(100, limit - len(rows)), offset=offset, return_meta=True)
        docs = page.get("documents") if isinstance(page, dict) else page
        docs = [d for d in (docs or []) if isinstance(d, dict)]
        if not docs:
            break
        rows.extend(docs)
        meta = page.get("meta") if isinstance(page, dict) else {}
        if not meta or not meta.get("has_more"):
            break
        offset = int(meta.get("next_offset") or (offset + len(docs)))
    return rows


def _category_filter(value: Any) -> str:
    text = _norm(value)
    aliases = {
        "tops": "top",
        "top": "top",
        "shirt": "top",
        "shirts": "top",
        "bottoms": "bottom",
        "bottom": "bottom",
        "pants": "bottom",
        "trousers": "bottom",
        "footwear": "footwear",
        "shoes": "footwear",
        "shoe": "footwear",
        "dresses": "dress",
        "dress": "dress",
        "outerwear": "outerwear",
        "accessories": "accessory",
        "accessory": "accessory",
    }
    return aliases.get(text, text)


def _select_p0_assets(
    rows: Iterable[Dict[str, Any]], *, limit: int = 0, category: str = ""
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    rows_by_role: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    wanted = _category_filter(category)
    for asset in rows:
        role = _asset_role(asset)
        if role in P0_TARGETS:
            rows_by_role[role].append(asset)
    role_order = (wanted,) if wanted in P0_TARGETS else ROLE_PRIORITY
    for role in role_order:
        for asset in rows_by_role.get(role, []):
            if _text(asset.get("board_image_url") or asset.get("boardImageUrl")):
                continue
            if counts[role] >= P0_TARGETS[role]:
                continue
            if not _source_url(asset):
                continue
            selected.append(asset)
            counts[role] += 1
            if limit and len(selected) >= limit:
                return selected
    return selected


def process_assets(*, apply: bool, scan_limit: int, limit: int = 0, category: str = "") -> Dict[str, Any]:
    proxy = AppwriteProxy()
    storage = R2Storage()
    selected = _select_p0_assets(_list_style_assets(scan_limit), limit=limit, category=category)
    selected_by_role = Counter(_asset_role(asset) for asset in selected)
    selected_by_gender = Counter(_norm(asset.get("gender") or "unknown") or "unknown" for asset in selected)
    samples = [
        {
            "asset_id": _doc_id(asset),
            "name": _text(asset.get("name")),
            "role": _asset_role(asset),
            "gender": _text(asset.get("gender") or "unknown"),
            "target_r2_key": _target_key(asset),
        }
        for asset in selected[:5]
    ]
    print(
        json.dumps(
            {
                "event": "AHVI_STYLE_ASSET_CUTOUT_SELECTED",
                "count": len(selected),
                "by_role": dict(selected_by_role),
                "by_gender": dict(selected_by_gender),
                "samples": samples,
                "dry_run": not apply,
            },
            sort_keys=True,
        )
    )
    stats: Dict[str, Any] = {
        "processed": 0,
        "skipped": 0,
        "ready": 0,
        "failed": 0,
        "by_role": defaultdict(int),
    }
    for asset in selected:
        role = _asset_role(asset)
        doc_id = _doc_id(asset)
        stats["processed"] += 1
        stats["by_role"][role] += 1
        try:
            raw = _download(_source_url(asset))
            cutout = remove_bg_external_sync(raw)
            png = _encode_png(cutout)
            ok, reason = validate_alpha_png(png)
            if not ok:
                raise RuntimeError(reason)
            key = _target_key(asset)
            public_url = f"dry-run://{key}"
            if apply:
                public_url = _upload_png(storage, key=key, image_bytes=png)
                proxy.update_document(
                    "style_assets",
                    doc_id,
                    {
                        "board_image_url": public_url,
                        "board_r2_key": key,
                        "cutout_status": "ready",
                        "catalog_image_url": _source_url(asset),
                    },
                )
            stats["ready"] += 1
            print(json.dumps({"event": "AHVI_STYLE_ASSET_CUTOUT_READY", "asset_id": doc_id, "role": role, "url": public_url}))
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            if apply and doc_id:
                try:
                    proxy.update_document("style_assets", doc_id, {"cutout_status": "failed"})
                except Exception:
                    pass
            print(json.dumps({"event": "AHVI_STYLE_ASSET_CUTOUT_FAILED", "asset_id": doc_id, "role": role, "reason": str(exc)[:160]}))
    stats["by_role"] = dict(stats["by_role"])
    print(
        "ahvi.style_asset_cutout_migration processed={processed} skipped={skipped} ready={ready} failed={failed} by_role={by_role}".format(
            **stats
        )
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate transparent board PNGs for P0 style assets.")
    parser.add_argument("--apply", action="store_true", help="Write R2/Appwrite updates. Default is dry-run.")
    parser.add_argument("--scan-limit", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0, help="Cap processed assets after P0 selection.")
    parser.add_argument("--category", default="", help="Optional role/category filter: tops, bottoms, footwear, dresses, outerwear, accessories.")
    args = parser.parse_args()
    stats = process_assets(
        apply=bool(args.apply),
        scan_limit=max(1, args.scan_limit),
        limit=max(0, args.limit),
        category=args.category,
    )
    print(json.dumps({"success": True, "dry_run": not args.apply, "stats": stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
