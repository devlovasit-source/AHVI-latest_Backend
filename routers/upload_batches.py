"""Sequential wardrobe upload batch endpoints (AHVI P0 upload MVP).

Client contract: one image per call, in order.

    POST /api/wardrobe/upload-batches                     -> create/resume batch
    POST /api/wardrobe/upload-batches/{batch_id}/items     -> process ONE image
    GET  /api/wardrobe/upload-batches/{batch_id}           -> poll batch status

This is additive: /analyze, /analyze-batch and /save-selected are untouched.
Authority for identity is always the authenticated request, never the body -
same contract as routers.style_boards / routers.wardrobe_capture.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from routers.wardrobe_capture import _effective_user_id
from services.upload_batch_orchestrator import UploadBatchInfraError, upload_batch_orchestrator

router = APIRouter(prefix="/api/wardrobe/upload-batches", tags=["wardrobe-upload-batch"])


class CreateBatchRequest(BaseModel):
    user_id: str = ""
    client_batch_request_id: str = Field(..., min_length=1)
    total_items: int = Field(..., ge=1, le=6)


class ProcessBatchItemRequest(BaseModel):
    user_id: str = ""
    client_upload_item_id: str = Field(..., min_length=1)
    image_base64: str = Field(..., min_length=20)
    metadata: Optional[Dict[str, Any]] = None
    # Explicit, item-scoped "Add anyway" - set only when the user picked
    # "Include Duplicate Item" for THIS specific image. Scoped by construction:
    # each call is one item, so this can never affect any other item's result.
    override_duplicate: bool = False
    # The exact garment the user already reviewed/approved in preview - the
    # same detected-item dict shape analyze_capture()/save-selected already
    # use (item_id, category, name, duplicate, raw_image_base64/masked_image_base64,
    # validation_status, etc). When present, save persists THIS item instead
    # of re-running detection on image_base64 from scratch.
    reviewed_item: Optional[Dict[str, Any]] = None


def _infra_failure(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


@router.post("")
def create_batch(http_request: Request, request: CreateBatchRequest) -> Dict[str, Any]:
    user_id = _effective_user_id(http_request, request.user_id)
    try:
        result = upload_batch_orchestrator.create_or_resume_batch(
            user_id=user_id,
            client_batch_request_id=request.client_batch_request_id,
            total_items=request.total_items,
        )
    except UploadBatchInfraError as exc:
        raise _infra_failure(exc) from exc
    if not result.get("success"):
        raise HTTPException(status_code=403, detail=result.get("reason", "batch_denied"))
    return {
        "success": True,
        "batch_id": result["batch_id"],
        "resumed": bool(result.get("resumed")),
    }


@router.post("/{batch_id}/items")
async def process_batch_item(
    batch_id: str, http_request: Request, request: ProcessBatchItemRequest
) -> Dict[str, Any]:
    user_id = _effective_user_id(http_request, request.user_id)

    # batch_id here is the caller-facing client_batch_request_id (same value
    # passed to create_batch) - the deterministic Appwrite doc id is an
    # internal detail derived from (user_id, client_batch_request_id), never
    # trusted from the URL directly.
    status = upload_batch_orchestrator.get_batch_status(user_id, batch_id)
    if not status.get("success"):
        if status.get("reason") == "unauthorized":
            raise HTTPException(status_code=404, detail="batch_not_found")
        raise HTTPException(status_code=404, detail="batch_not_found")

    try:
        result = await upload_batch_orchestrator.process_single_batch_item(
            http_request=http_request,
            user_id=user_id,
            batch_id=status["batch_id"],
            client_upload_item_id=request.client_upload_item_id,
            image_base64=request.image_base64,
            metadata=request.metadata or {},
            override_duplicate=request.override_duplicate,
            reviewed_item=request.reviewed_item,
        )
    except UploadBatchInfraError as exc:
        raise _infra_failure(exc) from exc

    return result


@router.get("/{batch_id}")
def get_batch(batch_id: str, http_request: Request, user_id: str = "") -> Dict[str, Any]:
    resolved_user_id = _effective_user_id(http_request, user_id)
    try:
        result = upload_batch_orchestrator.get_batch_status(resolved_user_id, batch_id)
    except UploadBatchInfraError as exc:
        raise _infra_failure(exc) from exc
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("reason", "batch_not_found"))
    return result
