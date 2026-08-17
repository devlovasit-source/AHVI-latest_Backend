import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from appwrite.query import Query
from appwrite.services.users import Users

from middleware.auth_middleware import get_current_user
from services.appwrite_service import get_admin_client


router = APIRouter(
    prefix="/auth",
    tags=["Auth"],
)


class PhoneLinkRequest(BaseModel):
    phone: str


def _normalize_phone(phone: str) -> str:
    """
    Normalize an E.164-style phone number without changing its country code.
    Example: +91 98765 43210 -> +919876543210
    """
    value = str(phone or "").strip()

    if not value:
        raise HTTPException(
            status_code=400,
            detail="Phone number is required",
        )

    # Remove common formatting characters.
    value = re.sub(r"[\s().-]", "", value)

    if not re.fullmatch(r"\+[1-9]\d{7,14}", value):
        raise HTTPException(
            status_code=400,
            detail="Invalid phone number format",
        )

    return value


def _get_user_id(user: Any) -> str:
    if isinstance(user, dict):
        return str(
            user.get("user_id")
            or user.get("$id")
            or user.get("id")
            or ""
        ).strip()

    for field in ("user_id", "$id", "id"):
        value = getattr(user, field, None)
        if value:
            return str(value).strip()

    return ""


@router.post("/phone/link")
def link_phone(
    payload: PhoneLinkRequest,
    current_user: Any = Depends(get_current_user),
):
    """
    Attach a phone number to the currently authenticated Appwrite user.

    Rules:
    - If nobody owns the phone, attach it to the current user.
    - If the current user already owns it, return success.
    - If another Appwrite user owns it, reject the request.
    - Never merge Appwrite accounts.
    """
    current_user_id = _get_user_id(current_user)

    if not current_user_id:
        raise HTTPException(
            status_code=401,
            detail="Authenticated user is required",
        )

    phone = _normalize_phone(payload.phone)

    users = Users(get_admin_client())

    try:
        result = users.list(
            queries=[
                Query.equal("phone", phone),
            ],
            total=True,
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Unable to verify phone ownership",
        )

    existing_users = getattr(result, "users", None)

    if existing_users is None and isinstance(result, dict):
        existing_users = result.get("users", [])

    existing_users = existing_users or []

    for existing_user in existing_users:
        existing_user_id = _get_user_id(existing_user)

        if existing_user_id == current_user_id:
            return {
                "status": "already_linked",
            }

        # The phone belongs to a different Appwrite account.
        raise HTTPException(
            status_code=409,
            detail="Phone number is already associated with another account",
        )

    try:
        users.update_phone(
            user_id=current_user_id,
            number=phone,
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Unable to link phone number",
        )

    return {
        "status": "linked",
    }