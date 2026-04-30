import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# =========================
# INIT
# =========================
device = torch.device("cpu")

processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
)

model.to(device)
model.eval()


# =========================
# PROMPT (IMPORTANT)
# =========================
TEXT_PROMPT = """
shirt . t-shirt . pants . jeans . dress .
saree . lehenga . kurta . dupatta .
watch . belt . bracelet . necklace . earring .
"""


# =========================
# DETECTION FUNCTION
# =========================
def detect_items_dino(image: Image.Image):

    inputs = processor(
        images=image,
        text=TEXT_PROMPT,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        threshold=0.35,
        target_sizes=[image.size[::-1]]
    )[0]

    detections = []

    for score, label, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"]
    ):
        detections.append({
            "label": processor.tokenizer.decode([label]).strip(),
            "bbox": box.tolist(),
            "score": float(score)
        })

    return detections
