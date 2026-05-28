"""Authenticated AHVI contacts API."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from models.ahvi_contact_models import (
    AhviContactCreate,
    AhviContactUpdate,
    contact_from_appwrite,
    contact_to_appwrite,
    model_to_dict,
)
from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError
from services.auth_helpers import require_user

router = APIRouter()
logger = logging.getLogger("ahvi.contacts")


def _is_owner(doc: Dict[str, Any], user_id: str) -> bool:
    owner = str(doc.get("userId") or doc.get("user_id") or "").strip()
    return bool(owner and owner == user_id)


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": {"code": code, "message": message}},
    )


def _proxy_error(exc: AppwriteProxyError) -> HTTPException:
    status_code = getattr(exc, "status_code", None) or 500
    if status_code == 404:
        return HTTPException(status_code=404, detail="Contact not found")
    return HTTPException(status_code=status_code, detail=str(exc) or "Contacts unavailable")


def _validate_contact_payload(payload: Dict[str, Any]) -> Optional[JSONResponse]:
    if not str(payload.get("firstname") or "").strip():
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            "CONTACT_NAME_REQUIRED",
            "Contact name is required.",
        )
    if not str(payload.get("phoneno") or "").strip():
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            "CONTACT_PHONE_REQUIRED",
            "Contact phone number is required.",
        )
    payload["phoneno"] = str(payload.get("phoneno") or "").strip()
    return None


def _load_owned(proxy: AppwriteProxy, contact_id: str, user_id: str) -> Dict[str, Any]:
    try:
        doc = proxy.get_document("contacts", contact_id)
    except AppwriteProxyError as exc:
        raise _proxy_error(exc)
    if not _is_owner(doc, user_id):
        raise HTTPException(status_code=404, detail="Contact not found")
    return doc


def _require_contacts_user(request: Request) -> str:
    has_auth = bool((request.headers.get("authorization") or "").strip())
    if not has_auth:
        logger.warning("contacts_auth_missing path=%s method=%s", request.url.path, request.method)
    return require_user(request)


@router.get("")
def list_contacts(
    request: Request,
    q: Optional[str] = Query(default=None, max_length=120),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    user_id = _require_contacts_user(request)
    logger.info("contacts_list_request user_id=%s q=%s limit=%s offset=%s", user_id, q or "", limit, offset)
    proxy = AppwriteProxy()
    try:
        logger.info("contacts_user_scoped_query user_id=%s resource=contacts", user_id)
        result = proxy.list_documents(
            "contacts", user_id=user_id, limit=limit, offset=offset, return_meta=True
        )
    except AppwriteProxyError as exc:
        raise _proxy_error(exc)

    contacts: List[Dict[str, Any]] = [
        contact_from_appwrite(doc)
        for doc in result.get("documents", [])
        if _is_owner(doc, user_id)
    ]
    query = (q or "").strip().lower()
    if query:
        contacts = [
            item
            for item in contacts
            if query in " ".join(
                [
                    str(item.get("displayName") or ""),
                    str(item.get("firstName") or ""),
                    str(item.get("lastName") or ""),
                    str(item.get("phoneNumber") or ""),
                    str(item.get("relationship") or ""),
                    " ".join(item.get("tags") or []),
                ]
            ).lower()
        ]
    contacts.sort(key=lambda item: (not bool(item.get("isFavorite")), str(item.get("displayName") or "").lower()))
    return {
        "success": True,
        "contacts": contacts,
        "items": contacts,
        "meta": result.get("meta") or {},
    }


@router.post("")
def create_contact(request: Request, contact: AhviContactCreate) -> Dict[str, Any]:
    user_id = _require_contacts_user(request)
    logger.info("contacts_create_request contacts.create.request user_id=%s", user_id)
    payload = contact_to_appwrite(model_to_dict(contact), user_id)
    validation = _validate_contact_payload(payload)
    if validation is not None:
        return validation
    try:
        created = AppwriteProxy().create_document("contacts", payload)
    except AppwriteProxyError as exc:
        logger.exception("contacts_create_error contacts.create.error user_id=%s error=%s", user_id, exc)
        return _error_response(
            getattr(exc, "status_code", None) or status.HTTP_502_BAD_GATEWAY,
            "CONTACT_CREATE_FAILED",
            str(exc) or "Could not save this contact.",
        )
    logger.info("contacts_create_success contacts.create.success user_id=%s contact_id=%s", user_id, created.get("$id") or created.get("id"))
    return {"success": True, "contact": contact_from_appwrite(created)}


@router.get("/{contact_id}")
def get_contact(request: Request, contact_id: str) -> Dict[str, Any]:
    user_id = _require_contacts_user(request)
    doc = _load_owned(AppwriteProxy(), contact_id, user_id)
    return {"success": True, "contact": contact_from_appwrite(doc)}


@router.put("/{contact_id}")
def update_contact(
    request: Request, contact_id: str, update: AhviContactUpdate
) -> Dict[str, Any]:
    user_id = _require_contacts_user(request)
    proxy = AppwriteProxy()
    _load_owned(proxy, contact_id, user_id)
    data = model_to_dict(update, exclude_unset=True)
    if not data:
        doc = _load_owned(proxy, contact_id, user_id)
        return {"success": True, "contact": contact_from_appwrite(doc)}
    payload = contact_to_appwrite(data, user_id)
    payload.pop("userId", None)
    if "phoneno" in payload:
        payload["phoneno"] = str(payload.get("phoneno") or "").strip()
    try:
        updated = proxy.update_document("contacts", contact_id, payload)
    except AppwriteProxyError as exc:
        logger.exception("contacts_update_error user_id=%s contact_id=%s error=%s", user_id, contact_id, exc)
        return _error_response(
            getattr(exc, "status_code", None) or status.HTTP_502_BAD_GATEWAY,
            "CONTACT_UPDATE_FAILED",
            str(exc) or "Could not update this contact.",
        )
    return {"success": True, "contact": contact_from_appwrite(updated)}


@router.patch("/{contact_id}")
def patch_contact(
    request: Request, contact_id: str, update: AhviContactUpdate
) -> Dict[str, Any]:
    return update_contact(request, contact_id, update)


@router.delete("/{contact_id}")
def delete_contact(request: Request, contact_id: str) -> Dict[str, Any]:
    user_id = _require_contacts_user(request)
    proxy = AppwriteProxy()
    _load_owned(proxy, contact_id, user_id)
    try:
        proxy.delete_document("contacts", contact_id)
    except AppwriteProxyError as exc:
        raise _proxy_error(exc)
    return {"success": True, "deleted": True, "id": contact_id}
