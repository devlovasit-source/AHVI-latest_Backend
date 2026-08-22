"""Bill scan endpoint — extracts bill fields from an uploaded receipt
image via the existing Ollama vision pipeline (the same one wardrobe
capture uses)."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from services.ai_gateway import ollama_vision_json
from services.auth_helpers import require_user

router = APIRouter(prefix="/api/bills", tags=["bills"])
logger = logging.getLogger("ahvi.routers.bills")


class BillScanRequest(BaseModel):
    image_base64: str = Field(..., min_length=10)


_SCAN_PROMPT = (
    "You are reading a bill, receipt, or invoice image. Extract the key "
    "fields and return ONLY a JSON object with this exact shape — no "
    "commentary, no markdown, no extra keys:\n"
    "{\n"
    '  "store": "<merchant or biller name; empty string if unclear>",\n'
    '  "amount": <total amount due as a number, no currency symbol; 0 if unclear>,\n'
    '  "date": "<due date or transaction date as YYYY-MM-DD; empty string if unclear>",\n'
    '  "category": "<one of: Rent, Utilities, Subscription, Shopping, Food, '
    'Transport, Healthcare, Insurance, Education, Entertainment, Other>",\n'
    '  "items": "<concise comma-separated list of line items; empty string if not visible>",\n'
    '  "currency": "<ISO 4217 currency code, e.g. INR, USD; empty if unclear>"\n'
    "}\n"
    "Return the object only."
)

_ALLOWED_CATEGORIES = {
    "Rent",
    "Utilities",
    "Subscription",
    "Shopping",
    "Food",
    "Transport",
    "Healthcare",
    "Insurance",
    "Education",
    "Entertainment",
    "Other",
}


def _coerce_amount(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().replace(",", "")
    # Strip leading currency symbols / letters.
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch in ".-")
    try:
        return float(cleaned) if cleaned else 0.0
    except Exception:
        return 0.0


def _normalize_category(value: Any) -> str:
    text = str(value or "").strip().title()
    if not text:
        return "Other"
    # Tolerate small variations.
    for cat in _ALLOWED_CATEGORIES:
        if cat.lower() == text.lower():
            return cat
    return "Other"


@router.post("/scan")
def scan_bill(request: Request, body: BillScanRequest) -> Dict[str, Any]:
    user_id = require_user(request)
    try:
        parsed, model = ollama_vision_json(
            prompt=_SCAN_PROMPT,
            image_base64=body.image_base64,
            timeout_seconds=60,
            usecase="bill_scan",
        )
    except Exception as exc:
        logger.warning("bill_scan failed user_id=%s error=%s", user_id, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Bill scan vision pipeline failed: {exc}",
        )

    if not isinstance(parsed, dict):
        parsed = {}

    extracted = {
        "store": str(parsed.get("store") or "").strip(),
        "amount": _coerce_amount(parsed.get("amount")),
        "date": str(parsed.get("date") or "").strip(),
        "category": _normalize_category(parsed.get("category")),
        "items": str(parsed.get("items") or "").strip(),
        "currency": str(parsed.get("currency") or "").strip().upper(),
    }
    logger.info(
        "bill_scan ok user_id=%s store=%r amount=%s category=%s model=%s",
        user_id,
        extracted["store"][:40],
        extracted["amount"],
        extracted["category"],
        model,
    )
    return {"success": True, "extracted": extracted, "model": model}
