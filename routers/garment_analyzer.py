
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
from transformers import pipeline
import io
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

# 🔥 NEW: image embedding
from services.image_embedding_service import encode_image_bytes

# =========================
# ROUTER
# =========================
router = APIRouter(
    prefix="/garment",
    tags=["Garment Analyzer"]
)

print("Loading Garment Classification Model (CLIP)...")
classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
print("Garment Model loaded successfully!")

# =========================
# CATEGORIES
# =========================
MAIN_CATEGORIES = [
    "traditional indian wear",
    "top wear",
    "bottom wear",
    "dresses and jumpsuits",
    "outerwear and winter wear",
    "footwear"
]

SUB_CATEGORIES = {
    "traditional indian wear": ["saree", "salwar kameez", "anarkali suit", "kurti", "lehenga choli", "sherwani", "dhoti"],
    "top wear": ["t-shirt", "polo shirt", "formal shirt", "casual shirt", "crop top", "peplum top", "tunic"],
    "bottom wear": ["skinny jeans", "straight jeans", "formal trousers", "cargo pants", "palazzo pants", "shorts", "skirt", "joggers"],
    "dresses and jumpsuits": ["maxi dress", "midi dress", "bodycon dress", "wrap dress", "full length jumpsuit", "romper"],
    "outerwear and winter wear": ["blazer", "denim jacket", "leather jacket", "trench coat", "woolen sweater", "hoodie", "cardigan", "coat"],
    "footwear": ["sneakers", "oxford formal shoes", "loafers", "high heels", "flat sandals", "boots", "slippers"]
}

APP_CATEGORY_MAP = {
    "top wear": "Tops",
    "bottom wear": "Bottoms",
    "footwear": "Footwear",
    "outerwear and winter wear": "Outerwear",
    "dresses and jumpsuits": "Dresses",
    "traditional indian wear": "Dresses"
}

# =========================
# COLOR EXTRACTION
# =========================
def get_dominant_color(cv_image, k=4):
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (120, 120), interpolation=cv2.INTER_AREA)

    pixels = image.reshape((-1, 3))

    filtered_pixels = [p for p in pixels if not (np.all(p < 30) or np.all(p > 230))]
    if not filtered_pixels:
        filtered_pixels = pixels

    filtered_pixels = np.array(filtered_pixels)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(filtered_pixels)

    counts = Counter(kmeans.labels_)
    dominant_cluster_index = counts.most_common(1)[0][0]
    dominant_rgb = [int(x) for x in kmeans.cluster_centers_[dominant_cluster_index]]

    hex_color = "#{:02x}{:02x}{:02x}".format(*dominant_rgb)
    return hex_color, dominant_rgb

# =========================
# MAIN API
# =========================
@router.post("/analyze/")
def analyze_garment(image_file: UploadFile = File(...)):
    try:
        contents = image_file.file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty image")

        # =========================
        # IMAGE LOAD
        # =========================
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        np_img = np.frombuffer(contents, np.uint8)
        cv_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if cv_image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # =========================
        # 🔥 IMAGE EMBEDDING (NEW)
        # =========================
        image_embedding = []
        try:
            image_embedding = encode_image_bytes(contents)
            if len(image_embedding) < 100:
                image_embedding = []  # safety
        except Exception:
            image_embedding = []

        # =========================
        # MAIN CATEGORY
        # =========================
        main_results = classifier(
            pil_image,
            candidate_labels=MAIN_CATEGORIES,
            hypothesis_template="a fashion catalog photo showing a {}"
        )

        winning_main_category = main_results[0]["label"]
        main_confidence = round(main_results[0]["score"], 4)

        # =========================
        # SUB CATEGORY
        # =========================
        specific_labels = SUB_CATEGORIES.get(winning_main_category, [])
        sub_results = classifier(
            pil_image,
            candidate_labels=specific_labels,
            hypothesis_template="a piece of clothing, specifically a {}, isolated on a background"
        )

        winning_sub_category = sub_results[0]["label"]
        sub_confidence = round(sub_results[0]["score"], 4)

        # =========================
        # COLOR
        # =========================
        hex_code, rgb_val = get_dominant_color(cv_image)

        # =========================
        # FINAL MAPPING
        # =========================
        item_name = winning_sub_category.title()
        mapped_app_category = APP_CATEGORY_MAP.get(winning_main_category, "Tops")

        return {
            "status": "success",
            "filename": image_file.filename,

            # classification
            "item_name": item_name,
            "main_category": winning_main_category,
            "app_category": mapped_app_category,
            "main_category_confidence": main_confidence,
            "sub_category": winning_sub_category,
            "sub_category_confidence": sub_confidence,

            # color
            "dominant_color_hex": hex_code,
            "dominant_color_rgb": rgb_val,

            # 🔥 NEW: embedding
            "image_embedding": image_embedding,

            # metadata
            "pipeline": "clip_with_embedding_v2"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[garment_analyzer] Error: {e}")
        raise HTTPException(status_code=500, detail="Garment analysis failed")
