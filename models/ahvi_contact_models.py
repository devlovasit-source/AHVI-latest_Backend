"""API models and Appwrite mappers for AHVI contacts."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AhviContactCreate(BaseModel):
    firstName: str = Field(..., min_length=1, max_length=80)
    lastName: Optional[str] = Field(default=None, max_length=80)
    phoneNumber: str = Field(..., min_length=3, max_length=40)
    displayName: Optional[str] = Field(default=None, max_length=160)
    relationship: Optional[str] = Field(default=None, max_length=80)
    notes: Optional[str] = Field(default=None, max_length=1000)
    tags: List[str] = Field(default_factory=list)
    isFavorite: bool = False
    avatarUrl: Optional[str] = Field(default=None, max_length=1000)


class AhviContactUpdate(BaseModel):
    firstName: Optional[str] = Field(default=None, min_length=1, max_length=80)
    lastName: Optional[str] = Field(default=None, max_length=80)
    phoneNumber: Optional[str] = Field(default=None, min_length=3, max_length=40)
    displayName: Optional[str] = Field(default=None, max_length=160)
    relationship: Optional[str] = Field(default=None, max_length=80)
    notes: Optional[str] = Field(default=None, max_length=1000)
    tags: Optional[List[str]] = None
    isFavorite: Optional[bool] = None
    avatarUrl: Optional[str] = Field(default=None, max_length=1000)


def model_to_dict(model: BaseModel, *, exclude_unset: bool = False) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_unset=exclude_unset)
    return model.dict(exclude_unset=exclude_unset)


def _clean_tags(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    tags: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in tags:
            tags.append(text[:40])
    return tags[:20]


def contact_to_appwrite(data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Map public API names to the existing Appwrite contact schema."""
    payload: Dict[str, Any] = {"userId": user_id}
    mapping = {
        "firstName": "firstname",
        "lastName": "surname",
        "phoneNumber": "phoneno",
        "displayName": "displayName",
        "relationship": "relationship",
        "notes": "notes",
        "isFavorite": "isFavorite",
        "avatarUrl": "avatarUrl",
    }
    for public_key, appwrite_key in mapping.items():
        if public_key not in data:
            continue
        value = data.get(public_key)
        if isinstance(value, str):
            value = value.strip()
        if value is not None:
            payload[appwrite_key] = value
    if "tags" in data:
        payload["tags"] = _clean_tags(data.get("tags"))
    if not payload.get("displayName"):
        parts = [payload.get("firstname"), payload.get("surname")]
        payload["displayName"] = " ".join(str(p).strip() for p in parts if p).strip()
    return payload


def contact_from_appwrite(doc: Dict[str, Any]) -> Dict[str, Any]:
    first = str(doc.get("firstname") or doc.get("firstName") or "").strip()
    last = str(doc.get("surname") or doc.get("lastName") or "").strip()
    display = str(doc.get("displayName") or f"{first} {last}".strip()).strip()
    return {
        "id": doc.get("$id") or doc.get("id"),
        "userId": doc.get("userId") or doc.get("user_id"),
        "firstName": first,
        "lastName": last,
        "phoneNumber": doc.get("phoneno") or doc.get("phoneNumber") or "",
        "displayName": display or first,
        "relationship": doc.get("relationship") or "",
        "notes": doc.get("notes") or "",
        "tags": _clean_tags(doc.get("tags")),
        "isFavorite": bool(doc.get("isFavorite") or False),
        "avatarUrl": doc.get("avatarUrl") or "",
        "createdAt": doc.get("$createdAt"),
        "updatedAt": doc.get("$updatedAt"),
    }
