import io
import uuid
import asyncio
import os
from typing import List, Dict

import numpy as np
from PIL import Image
import httpx

from services.bg_service import remove_bg_bytes
from services.r2_storage import R2Storage


# =========================
# CONFIG
# =========================
MAX_ITEMS = 5
RESIZE_LIMIT = 640

HF_TOKEN = os.getenv("HF_TOKEN")
HF_URL = "https://api-inference.huggingface.co/models/IDEA-Research/grounding-dino-tiny"

TEXT_PROMPT = "shirt . pants . dress . saree . kurta . watch . shoes . bag"

# 🔥 FEATURE FLAGS
ENABLE_DETECTION = os.getenv("ENABLE_DETECTION", "true") == "true"
ENABLE_MEDIAPIPE = os.getenv("ENABLE_MEDIAPIPE", "false") == "true"


# =========================
# LAZY MEDIAPIPE
# =========================
def get_mediapipe():
    if not ENABLE_MEDIAPIPE:
        return None

    try:
        import mediapipe as mp
        return mp
    except Exception as e:
        print("[mediapipe disabled]", e)
        return None


# =========================
# RESIZE
# =========================
def resize_image(image: Image.Image):
    w, h = image.size
    if max(w, h) <= RESIZE_LIMIT:
        return image

    scale = RESIZE_LIMIT / max(w, h)
    return image.resize((int(w * scale), int(h * scale)))


def image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


# =========================
# HF DETECTION (ASYNC)
# =========================
async def hf_detect_async(image: Image.Image):
    if not HF_TOKEN or not ENABLE_DETECTION:
        return []

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    image_bytes = image_to_bytes(image)

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            res = await client.post(
                HF_URL,
                headers=headers,
                content=image_bytes,
                params={"text": TEXT_PROMPT},
            )

        if res.status_code != 200:
            print("[HF ERROR]", res.text)
            return []

        return res.json()

    except Exception as e:
        print("[HF EXCEPTION]", e)
        return []


# =========================
# PARSER
# =========================
def parse_detections(hf_output):
    if not hf_output:
        return []

    if isinstance(hf_output, dict):
        if "error" in hf_output:
            return []
        hf_output = hf_output.get("outputs", [])

    detections = []

    for item in hf_output:
        box = item.get("box", {})

        detections.append({
            "label": item.get("label", "item"),
            "bbox": [
                int(box.get("xmin", 0)),
                int(box.get("ymin", 0)),
                int(box.get("xmax", 0)),
                int(box.get("ymax", 0)),
            ],
            "score": float(item.get("score", 0))
        })

    return detections


# =========================
# MEDIAPIPE SAFE
# =========================
def get_regions_safe(image_np):
    mp = get_mediapipe()
    if not mp:
        return []

    regions = []
    h, w, _ = image_np.shape

    try:
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as face:
            res = face.process(image_np)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0]
                x = int(lm.landmark[234].x * w)
                y = int(lm.landmark[234].y * h)
                regions.append({
                    "label": "earring",
                    "bbox": [x-40, y-40, x+40, y+40],
                    "score": 0.9
                })

        with mp.solutions.pose.Pose(static_image_mode=True) as pose:
            res = pose.process(image_np)
            if res.pose_landmarks:
                wrist = res.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
                x = int(wrist.x * w)
                y = int(wrist.y * h)
                regions.append({
                    "label": "watch",
                    "bbox": [x-50, y-50, x+50, y+50],
                    "score": 0.9
                })

    except Exception as e:
        print("[mediapipe runtime error]", e)

    return regions


# =========================
# IOU + FILTER
# =========================
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((box1[2]-box1[0])*(box1[3]-box1[1]) +
             (box2[2]-box2[0])*(box2[3]-box2[1]) - inter)

    return inter / union if union else 0


def filter_and_limit(detections, width, height):
    filtered = []

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]

        if (x2 - x1) < 30 or (y2 - y1) < 30:
            continue

        area = (x2 - x1) * (y2 - y1)
        if area > 0.9 * width * height:
            continue

        if any(iou(f["bbox"], d["bbox"]) > 0.7 for f in filtered):
            continue

        filtered.append(d)

    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered[:MAX_ITEMS]


# =========================
# BG REMOVAL
# =========================
async def batch_bg(crops):
    return await asyncio.gather(
        *[remove_bg_bytes(c) for c in crops]
    )


# =========================
# MAIN PIPELINE
# =========================
async def run_hybrid_detection(image: Image.Image):
    image = resize_image(image)

    width, height = image.size
    image_np = np.array(image)

    r2 = R2Storage()

    # -------------------------
    # DETECTION (OPTIONAL)
    # -------------------------
    detections = []

    try:
        hf_output = await hf_detect_async(image)
        detections = parse_detections(hf_output)
    except Exception as e:
        print("[HF DETECTION FAILED]", e)

    # -------------------------
    # MEDIAPIPE (OPTIONAL)
    # -------------------------
    try:
        detections.extend(get_regions_safe(image_np))
    except Exception:
        pass

    # -------------------------
    # FALLBACK (IMPORTANT 🔥)
    # -------------------------
    if not detections:
        detections = [{
            "label": "item",
            "bbox": [0, 0, width, height],
            "score": 1.0
        }]

    detections = filter_and_limit(detections, width, height)

    # -------------------------
    # CROPS
    # -------------------------
    crops = []
    meta = []

    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        crop = image.crop((x1, y1, x2, y2))

        buf = io.BytesIO()
        crop.save(buf, format="JPEG")

        crops.append(buf.getvalue())
        meta.append(d)

    # -------------------------
    # BG REMOVAL
    # -------------------------
    masked_list = await batch_bg(crops)

    # -------------------------
    # UPLOAD
    # -------------------------
    async def upload_one(raw, masked, label):
        file_id = str(uuid.uuid4())

        upload = r2.upload_wardrobe_images(
            file_id=file_id,
            raw_image_bytes=raw,
            masked_image_bytes=masked
        )

        return {
            "item_id": file_id,
            "label": label,
            "raw_url": upload["raw_image_url"],
            "masked_url": upload["masked_image_url"]
        }

    results = await asyncio.gather(
        *[
            upload_one(crops[i], masked_list[i], meta[i]["label"])
            for i in range(len(crops))
        ]
    )

    return results
