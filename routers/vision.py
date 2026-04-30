import base64
import io
import asyncio
import hashlib
import json

import httpx
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from PIL import Image

from services.embedding_service import encode_metadata
from services.image_embedding_service import encode_image_url
from services.qdrant_service import qdrant_service
from services import ai_gateway
from services.auth_service import get_current_user
from services.security_limits import get_redis_client


router = APIRouter()

# 🔥 CONFIG
RUNPOD_URL = "https://your-runpod-endpoint"  # replace
AI_CACHE_TTL = 86400
SEM = asyncio.Semaphore(5)


# =========================
# REQUEST
# =========================
class ImageAnalyzeRequest(BaseModel):
    image_base64: str = Field(..., min_length=20)


# =========================
# HELPERS
# =========================
def _decode_image(value: str) -> Image.Image:
    try:
        if "," in value:
            value = value.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")


# =========================
# 🔴 REDIS AI CACHE
# =========================
def _ai_key(url: str):
    return "ai:" + hashlib.md5(url.encode()).hexdigest()


async def _get_ai(masked_url: str):
    redis = await get_redis_client()
    key = _ai_key(masked_url)

    if redis:
        try:
            cached = await redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

    loop = asyncio.get_running_loop()

    try:
        ai, _ = await loop.run_in_executor(
            None,
            lambda: ai_gateway.ollama_vision_json(
                prompt="Describe fashion attributes",
                image_base64=masked_url,
                usecase="vision",
            ),
        )
    except Exception:
        ai = {}

    if redis and ai:
        try:
            await redis.setex(key, AI_CACHE_TTL, json.dumps(ai))
        except Exception:
            pass

    return ai


# =========================
# ⚡ GPU DETECTION (RUNPOD)
# =========================
async def _detect(image: Image.Image):
    try:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")

        async with httpx.AsyncClient(timeout=20) as client:
            res = await client.post(
                RUNPOD_URL,
                files={"file": buf.getvalue()},
            )
            return res.json()
    except Exception as e:
        print("[DETECTION ERROR]", e)
        return []


# =========================
# 🧠 BATCH EMBEDDINGS
# =========================
async def _batch_embeddings(items):
    loop = asyncio.get_running_loop()

    text_tasks = [
        loop.run_in_executor(None, encode_metadata, i)
        for i in items
    ]

    image_tasks = [
        loop.run_in_executor(None, encode_image_url, i["masked_url"])
        for i in items
    ]

    tvs = await asyncio.gather(*text_tasks)
    ivs = await asyncio.gather(*image_tasks)

    vectors = []

    for tv, iv in zip(tvs, ivs):
        if tv and iv:
            vec = [(0.6*i + 0.4*t) for i, t in zip(iv, tv)]
        else:
            vec = iv or tv or []
        vectors.append(vec)

    return vectors


# =========================
# PROCESS PIPELINE
# =========================
async def process_items(image: Image.Image, user_id: str):

    detections = await _detect(image)

    if not detections:
        return {"items": [], "meta": {"detection": "failed"}}

    # prepare items
    items = []
    for d in detections:
        items.append({
            "category": d.get("label"),
            "image_url": d.get("raw_url"),
            "masked_url": d.get("masked_url"),
            "user_id": user_id,
            "image_id": d.get("item_id"),
        })

    # 🔴 AI (parallel + cached)
    async def enrich(item):
        if item["category"] not in ["ring", "earring", "bracelet"]:
            ai = await _get_ai(item["masked_url"])
            item.update({
                "pattern": ai.get("pattern"),
                "occasions": ai.get("occasions"),
                "style": ai.get("style"),
            })
        return item

    items = await asyncio.gather(*[enrich(i) for i in items])

    # 🧠 batch embeddings
    vectors = await _batch_embeddings(items)

    results = []

    for item, vector in zip(items, vectors):
        similar = qdrant_service.search_similar(vector, user_id, 5) if vector else []

        if vector:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: qdrant_service.upsert_item(
                    item_id=item["image_id"],
                    vector=vector,
                    payload=item,
                ),
            )

        results.append({
            "item_id": item["image_id"],
            "category": item["category"],
            "image_url": item["image_url"],
            "masked_url": item["masked_url"],
            "similar_items": similar,
        })

    return {
        "items": results,
        "meta": {"detection": "success"}
    }


# =========================
# ROUTE
# =========================
@router.post("/analyze-image")
async def analyze_image(
    request: ImageAnalyzeRequest,
    user=Depends(get_current_user),
):
    image = _decode_image(request.image_base64)

    result = await process_items(image, user["user_id"])

    return {
        "success": True,
        "total_items": len(result["items"]),
        "items": result["items"],
        "meta": result["meta"],
    }
