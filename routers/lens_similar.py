from __future__ import annotations

import logging
import os
from urllib.parse import urlparse
from typing import Any, Dict, List

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/api/lens", tags=["lens"])
logger = logging.getLogger("ahvi.lens_similar")


def _source_from_url(value: str) -> str:
    try:
        host = urlparse(value).netloc.replace("www.", "")
        return host or ""
    except Exception:
        return ""


async def upload_image_for_lens(file: UploadFile) -> str:
    upload_url = os.getenv("CLOUDINARY_UPLOAD_URL", "").strip()
    upload_preset = os.getenv("CLOUDINARY_UPLOAD_PRESET", "").strip()
    if not upload_url or not upload_preset:
        raise HTTPException(
            status_code=503,
            detail="Lens image upload is not configured.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty.")

    timeout = httpx.Timeout(25.0, connect=8.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            upload_url,
            data={"upload_preset": upload_preset},
            files={
                "file": (
                    file.filename or "ahvi-lens.jpg",
                    content,
                    file.content_type or "image/jpeg",
                )
            },
        )
    if response.status_code >= 400:
        logger.warning("lens_cloudinary_upload_failed status=%s body=%s", response.status_code, response.text[:240])
        raise HTTPException(status_code=502, detail="Could not upload image for Lens search.")
    payload = response.json()
    image_url = str(payload.get("secure_url") or payload.get("url") or "").strip()
    if not image_url:
        raise HTTPException(status_code=502, detail="Image host did not return a URL.")
    return image_url


def _normalize_match(raw: Dict[str, Any]) -> Dict[str, str] | None:
    title = str(raw.get("title") or raw.get("name") or "").strip()
    product_url = str(
        raw.get("link") or raw.get("product_link") or raw.get("source_link") or raw.get("url") or ""
    ).strip()
    image_url = str(
        raw.get("thumbnail")
        or raw.get("image")
        or raw.get("imageUrl")
        or raw.get("source_image")
        or raw.get("serpapi_thumbnail")
        or ""
    ).strip()
    if not title or (not product_url and not image_url):
        return None
    source = str(raw.get("source") or raw.get("domain") or _source_from_url(product_url) or "").strip()
    brand = str(raw.get("brand") or raw.get("merchant") or source or "").strip()
    return {
        "title": title,
        "brand": brand,
        "imageUrl": image_url,
        "productUrl": product_url,
        "source": source,
    }


async def search_google_lens_similar(image_url: str) -> List[Dict[str, str]]:
    api_key = os.getenv("SERPAPI_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="Google Lens search is not configured.")

    timeout = httpx.Timeout(25.0, connect=8.0)
    params = {"engine": "google_lens", "url": image_url, "api_key": api_key}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get("https://serpapi.com/search.json", params=params)
    if response.status_code >= 400:
        logger.warning("lens_serpapi_failed status=%s body=%s", response.status_code, response.text[:240])
        raise HTTPException(status_code=502, detail="Google Lens search failed.")

    payload = response.json()
    raw_results: List[Dict[str, Any]] = []
    for key in ("visual_matches", "shopping_results", "organic_results", "image_results"):
        value = payload.get(key)
        if isinstance(value, list):
            raw_results.extend(v for v in value if isinstance(v, dict))

    matches: List[Dict[str, str]] = []
    seen: set[str] = set()
    for raw in raw_results:
        normalized = _normalize_match(raw)
        if not normalized:
            continue
        dedupe_key = normalized["productUrl"] or normalized["imageUrl"]
        if not dedupe_key or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        matches.append(normalized)
        if len(matches) >= 10:
            break
    return matches


@router.post("/find-similar")
async def find_similar_items(file: UploadFile = File(...)):
    image_url = await upload_image_for_lens(file)
    matches = await search_google_lens_similar(image_url)
    return {
        "success": True,
        "source": "google_lens",
        "matches": matches,
        "message": (
            "I found similar products for this image."
            if matches
            else "I couldn't find similar products for this image yet."
        ),
    }
