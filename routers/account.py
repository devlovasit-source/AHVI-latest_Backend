"""Account management router: 45-day soft-delete lifecycle.

Endpoints:
- POST /api/account/delete: Request account deletion with 45-day grace period.
- POST /api/account/cancel-delete: Cancel pending deletion and reactivate account.
- GET  /api/account/status: Retrieve current deletion state and remaining days.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from services.auth_helpers import require_user
from services.account_deletion_service import (
    request_account_deletion,
    cancel_account_deletion,
    get_account_deletion_status,
)

logger = logging.getLogger("ahvi.routers.account")

router = APIRouter(prefix="/account", tags=["Account"])


class AccountDeleteRequest(BaseModel):
    confirmation: str = Field(
        ...,
        description="Confirmation token to prevent accidental deletion. Must be 'DELETE' or 'CONFIRM'.",
        min_length=1,
        max_length=20,
    )
    reason: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional reason for account deletion.",
    )


@router.post("/delete")
def handle_delete_account(
    req: AccountDeleteRequest,
    http_request: Request,
) -> Dict[str, Any]:
    """Request soft deletion of the authenticated account.

    Sets the account status to 'pending_deletion' with a 45-day grace period
    and revokes all active Appwrite user sessions. Requires confirmation token
    'DELETE' or 'CONFIRM'.
    """
    user_id = require_user(http_request)

    confirmation_token = str(req.confirmation or "").strip().upper()
    if confirmation_token not in {"DELETE", "CONFIRM"}:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. To schedule deletion, set confirmation='DELETE' or 'CONFIRM'.",
        )

    try:
        result = request_account_deletion(user_id=user_id, reason=req.reason)
        return result
    except Exception as exc:
        logger.exception("Error scheduling account deletion for user_id=%s: %s", user_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to schedule account deletion: {exc}") from exc


@router.post("/cancel-delete")
def handle_cancel_delete(
    http_request: Request,
) -> Dict[str, Any]:
    """Cancel a pending account deletion within the 45-day grace period.

    Restores account_status to 'active' and clears all deletion schedule timestamps.
    """
    user_id = require_user(http_request)

    status = get_account_deletion_status(user_id=user_id)
    if status.get("account_status") != "pending_deletion":
        return {
            "success": True,
            "user_id": user_id,
            "account_status": "active",
            "message": "Account is already active. No pending deletion to cancel.",
        }

    if status.get("is_expired"):
        raise HTTPException(
            status_code=410,
            detail="The 45-day grace period has expired. This account cannot be restored.",
        )

    try:
        result = cancel_account_deletion(user_id=user_id)
        return result
    except Exception as exc:
        logger.exception("Error cancelling account deletion for user_id=%s: %s", user_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to cancel account deletion: {exc}") from exc


@router.get("/status")
def handle_account_status(
    http_request: Request,
) -> Dict[str, Any]:
    """Check the deletion and activity status of the authenticated account."""
    user_id = require_user(http_request)

    try:
        status = get_account_deletion_status(user_id=user_id)
        return {
            "success": True,
            **status,
        }
    except Exception as exc:
        logger.exception("Error checking account status for user_id=%s: %s", user_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve account status: {exc}") from exc
