from typing import Any, Dict

from services.appwrite_proxy import AppwriteProxy


def create_document(
    *, resource: str, payload: Dict[str, Any], document_id: str = "unique()"
):
    return AppwriteProxy().create_document(resource, payload, document_id=document_id)


def update_document(*, resource: str, document_id: str, payload: Dict[str, Any]):
    return AppwriteProxy().update_document(resource, document_id, payload)


def delete_document(*, resource: str, document_id: str):
    AppwriteProxy().delete_document(resource, document_id)


def upsert_user_profile(*, user_id: str, payload: Dict[str, Any]):
    proxy = AppwriteProxy()
    try:
        return proxy.update_document("users", user_id, payload)
    except Exception:
        return proxy.create_document("users", payload, document_id=user_id)


# ================= AHVI STYLE PROFILE PATCH V2 BEGIN =================


def _ahvi_strip_appwrite_meta(doc):
    return {
        k: v
        for k, v in (doc or {}).items()
        if isinstance(k, str) and not k.startswith("$")
    }


def _ahvi_deep_merge(base, override):
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _ahvi_deep_merge(merged[key], value)
        elif value not in (None, ""):
            merged[key] = value
    return merged


def merge_user_profiles(*profiles):
    merged = {}
    for profile in profiles:
        if isinstance(profile, dict):
            merged = _ahvi_deep_merge(merged, _ahvi_strip_appwrite_meta(profile))
    return merged


def get_user_profile(*, user_id):
    uid = str(user_id or "").strip()
    if not uid:
        return {}

    try:
        doc = AppwriteProxy().get_document("users", uid)
        return _ahvi_strip_appwrite_meta(doc) if isinstance(doc, dict) else {}
    except Exception:
        pass

    try:
        docs = AppwriteProxy().list_documents("users", user_id=uid, limit=1)
        rows = (
            docs.get("documents") or docs.get("items") or []
            if isinstance(docs, dict)
            else docs or []
        )
        if rows and isinstance(rows[0], dict):
            return _ahvi_strip_appwrite_meta(rows[0])
    except Exception:
        pass

    return {}


# ================= AHVI STYLE PROFILE PATCH V2 END =================
