"""Authenticated AHVI contacts API."""
from __future__ import annotations

import logging
import re
import uuid
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
    if owner:
        return owner == user_id
    doc_id = str(doc.get("$id") or doc.get("id") or "").strip()
    return bool(doc_id and doc_id.startswith(_user_doc_prefix(user_id)))


def _user_doc_prefix(user_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]", "", str(user_id or ""))[:12]
    return f"ct_{safe or 'user'}_"


def _new_contact_doc_id(user_id: str) -> str:
    return f"{_user_doc_prefix(user_id)}{uuid.uuid4().hex[:12]}"


def _extract_unknown_attribute(exc: Exception) -> Optional[str]:
    match = re.search(r'Unknown attribute:\s*\\"([^"]+)\\"', str(exc))
    if match:
        return match.group(1)
    match = re.search(r'Unknown attribute:\s*"([^"]+)"', str(exc))
    return match.group(1) if match else None


def _extract_missing_attribute(exc: Exception) -> Optional[str]:
    match = re.search(r'Missing required attribute:\s*\\"([^"]+)\\"', str(exc))
    if match:
        return match.group(1)
    match = re.search(r'Missing required attribute:\s*"([^"]+)"', str(exc))
    return match.group(1) if match else None


def _placeholder_email(payload: Dict[str, Any]) -> str:
    phone = re.sub(r"[^0-9A-Za-z]", "", str(payload.get("phoneNumber") or payload.get("phoneno") or "contact"))
    return f"{phone or 'contact'}@ahvi.local"


def _create_contact_schema_adaptive(proxy: AppwriteProxy, payload: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Create contacts against either current AHVI schema or older dev schema."""
    data = dict(payload)
    document_id = _new_contact_doc_id(user_id)
    for _ in range(12):
        try:
            return proxy.create_document("contacts", data, document_id=document_id)
        except AppwriteProxyError as exc:
            unknown = _extract_unknown_attribute(exc)
            if unknown and unknown in data:
                logger.info("contacts_schema_strip_unknown user_id=%s attr=%s", user_id, unknown)
                data.pop(unknown, None)
                continue
            missing = _extract_missing_attribute(exc)
            if missing == "email" and not data.get("email"):
                logger.info("contacts_schema_add_placeholder_email user_id=%s", user_id)
                data["email"] = _placeholder_email(data)
                continue
            raise
    raise AppwriteProxyError(502, "Could not adapt contact payload to Appwrite schema")


def _update_contact_schema_adaptive(proxy: AppwriteProxy, contact_id: str, payload: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    data = dict(payload)
    for _ in range(12):
        try:
            return proxy.update_document("contacts", contact_id, data)
        except AppwriteProxyError as exc:
            unknown = _extract_unknown_attribute(exc)
            if unknown and unknown in data:
                logger.info("contacts_update_schema_strip_unknown user_id=%s attr=%s", user_id, unknown)
                data.pop(unknown, None)
                continue
            raise
    raise AppwriteProxyError(502, "Could not adapt contact update to Appwrite schema")


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
    if not str(payload.get("firstName") or payload.get("firstname") or "").strip():
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            "CONTACT_NAME_REQUIRED",
            "Contact name is required.",
        )
    if not str(payload.get("phoneNumber") or payload.get("phoneno") or "").strip():
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            "CONTACT_PHONE_REQUIRED",
            "Contact phone number is required.",
        )
    payload["phoneNumber"] = str(payload.get("phoneNumber") or payload.get("phoneno") or "").strip()
    payload.pop("phoneno", None)
    if not str(payload.get("lastName") or "").strip():
        payload["lastName"] = "-"
    if not str(payload.get("email") or "").strip():
        payload["email"] = _placeholder_email(payload)
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
        created = _create_contact_schema_adaptive(AppwriteProxy(), payload, user_id)
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
    if "phoneNumber" in payload:
        payload["phoneNumber"] = str(payload.get("phoneNumber") or "").strip()
    payload.pop("phoneno", None)
    try:
        updated = _update_contact_schema_adaptive(proxy, contact_id, payload, user_id)
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
