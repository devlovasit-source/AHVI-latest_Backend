"""Canonical Wardrobe Item Interaction Router.

Exposes REST APIs for wardrobe item lifecycle (Favorite, Wear Event / Wear History,
Wear Reminders, Care Rules, Soft Delete). All endpoints validate ownership
and return 404 on unowned or missing items.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from services.auth_helpers import enforce_owner
from services.wardrobe_item_service import WardrobeItemService
from services.wear_event_service import WearEventService
from services.wear_projection_service import WearProjectionService
from services.wardrobe_reminder_service import WardrobeReminderService
from services.wardrobe_care_service import WardrobeCareService

logger = logging.getLogger("ahvi.routers.wardrobe_interactions")

router = APIRouter(prefix="/api/wardrobe/items", tags=["wardrobe-interactions"])

_item_service = WardrobeItemService()
_wear_event_service = WearEventService()
_wear_projection_service = WearProjectionService()
_reminder_service = WardrobeReminderService()
_care_service = WardrobeCareService()


# ---------------------------------------------------------------------------
# DTO Models
# ---------------------------------------------------------------------------
class FavoriteRequest(BaseModel):
    user_id: Optional[str] = None
    is_favorite: bool = True


class RecordWearRequest(BaseModel):
    user_id: Optional[str] = None
    source: str = "wardrobe_item"
    occurred_at: str = ""
    timezone: str = "Asia/Kolkata"
    outfit_id: Optional[str] = None
    board_id: Optional[str] = None


class WearReminderRequest(BaseModel):
    user_id: Optional[str] = None
    scheduled_at: str
    timezone: str = "Asia/Kolkata"
    message: str = ""


class PatchWearReminderRequest(BaseModel):
    user_id: Optional[str] = None
    scheduled_at: Optional[str] = None
    status: Optional[str] = None
    message: Optional[str] = None


class CareRuleRequest(BaseModel):
    user_id: Optional[str] = None
    care_type: str = "wash"
    trigger_type: str = "date"
    scheduled_at: Optional[str] = None
    repeat_every_wears: Optional[int] = None


class PatchCareRuleRequest(BaseModel):
    user_id: Optional[str] = None
    care_type: Optional[str] = None
    scheduled_at: Optional[str] = None
    status: Optional[str] = None


# ---------------------------------------------------------------------------
# 1. Favorite Endpoint
# ---------------------------------------------------------------------------
@router.put("/{item_id}/favorite")
def set_favorite(item_id: str, req: Request, payload: FavoriteRequest):
    user_id = enforce_owner(req, payload.user_id)
    try:
        res = _item_service.set_favorite(
            user_id=user_id,
            item_id=item_id,
            is_favorite=payload.is_favorite,
        )
        return {"success": True, **res}
    except KeyError:
        raise HTTPException(status_code=404, detail="Item not found")
    except Exception as exc:
        logger.warning("ahvi.set_favorite_failed item_id=%s err=%s", item_id, str(exc)[:140])
        raise HTTPException(status_code=500, detail="Failed to update favorite status")


# ---------------------------------------------------------------------------
# 2. Wear Event & Wear History Endpoints
# ---------------------------------------------------------------------------
@router.post("/{item_id}/wear")
def record_item_wear(item_id: str, req: Request, payload: RecordWearRequest):
    user_id = enforce_owner(req, payload.user_id)
    try:
        _item_service.require_owned_item(user_id, item_id)
        res = _wear_event_service.record_wear(
            user_id=user_id,
            item_ids=[item_id],
            source=payload.source,
            entity_type="item",
            entity_id=item_id,
            outfit_id=payload.outfit_id,
            board_id=payload.board_id,
            occurred_at=payload.occurred_at,
            timezone=payload.timezone,
        )
        return {"success": True, **res}
    except KeyError:
        raise HTTPException(status_code=404, detail="Item not found")
    except Exception as exc:
        logger.warning("ahvi.record_item_wear_failed item_id=%s err=%s", item_id, str(exc)[:140])
        raise HTTPException(status_code=500, detail="Failed to record wear event")


@router.get("/{item_id}/wear-history")
def get_wear_history(item_id: str, req: Request, user_id: Optional[str] = None):
    uid = enforce_owner(req, user_id)
    try:
        _item_service.require_owned_item(uid, item_id)
        projection = _wear_projection_service.get_item_wear_history_projection(user_id=uid, item_id=item_id)
        return {"success": True, **projection}
    except KeyError:
        raise HTTPException(status_code=404, detail="Item not found")
    except Exception as exc:
        logger.warning("ahvi.get_wear_history_failed item_id=%s err=%s", item_id, str(exc)[:140])
        raise HTTPException(status_code=500, detail="Failed to fetch wear history")


# ---------------------------------------------------------------------------
# 3. Wear Reminders Endpoints
# ---------------------------------------------------------------------------
@router.post("/{item_id}/wear-reminders")
def schedule_wear_reminder(item_id: str, req: Request, payload: WearReminderRequest):
    user_id = enforce_owner(req, payload.user_id)
    try:
        res = _reminder_service.schedule_wear_reminder(
            user_id=user_id,
            item_id=item_id,
            scheduled_at=payload.scheduled_at,
            timezone=payload.timezone,
            message=payload.message,
        )
        return {"success": True, **res}
    except KeyError:
        raise HTTPException(status_code=404, detail="Item not found")
    except Exception as exc:
        logger.warning("ahvi.schedule_wear_reminder_failed item_id=%s err=%s", item_id, str(exc)[:140])
        raise HTTPException(status_code=500, detail="Failed to schedule wear reminder")


@router.get("/{item_id}/wear-reminders")
def list_wear_reminders(item_id: str, req: Request, user_id: Optional[str] = None):
    uid = enforce_owner(req, user_id)
    try:
        reminders = _reminder_service.list_wear_reminders(user_id=uid, item_id=item_id)
        return {"success": True, "reminders": reminders}
    except KeyError:
        raise HTTPException(status_code=404, detail="Item not found")


@router.patch("/{item_id}/wear-reminders/{reminder_id}")
def patch_wear_reminder(item_id: str, reminder_id: str, req: Request, payload: PatchWearReminderRequest):
    user_id = enforce_owner(req, payload.user_id)
    try:
        res = _reminder_service.patch_wear_reminder(
            user_id=user_id,
            item_id=item_id,
            reminder_id=reminder_id,
            scheduled_at=payload.scheduled_at,
            status=payload.status,
            message=payload.message,
        )
        return {"success": True, **res}
    except KeyError:
        raise HTTPException(status_code=404, detail="Reminder or item not found")


@router.delete("/{item_id}/wear-reminders/{reminder_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_wear_reminder(item_id: str, reminder_id: str, req: Request, user_id: Optional[str] = None):
    uid = enforce_owner(req, user_id)
    try:
        _reminder_service.delete_wear_reminder(user_id=uid, item_id=item_id, reminder_id=reminder_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except KeyError:
        raise HTTPException(status_code=404, detail="Reminder or item not found")


# ---------------------------------------------------------------------------
# 4. Care Rules Endpoints
# ---------------------------------------------------------------------------
@router.post("/{item_id}/care-rules")
def create_care_rule(item_id: str, req: Request, payload: CareRuleRequest):
    user_id = enforce_owner(req, payload.user_id)
    try:
        res = _care_service.create_care_rule(
            user_id=user_id,
            item_id=item_id,
            care_type=payload.care_type,
            trigger_type=payload.trigger_type,
            scheduled_at=payload.scheduled_at,
            repeat_every_wears=payload.repeat_every_wears,
        )
        return {"success": True, **res}
    except KeyError:
        raise HTTPException(status_code=404, detail="Item not found")


@router.get("/{item_id}/care-rules")
def list_care_rules(item_id: str, req: Request, user_id: Optional[str] = None):
    uid = enforce_owner(req, user_id)
    try:
        rules = _care_service.list_care_rules(user_id=uid, item_id=item_id)
        return {"success": True, "rules": rules}
    except KeyError:
        raise HTTPException(status_code=404, detail="Item not found")


@router.patch("/{item_id}/care-rules/{rule_id}")
def patch_care_rule(item_id: str, rule_id: str, req: Request, payload: PatchCareRuleRequest):
    user_id = enforce_owner(req, payload.user_id)
    try:
        res = _care_service.patch_care_rule(
            user_id=user_id,
            item_id=item_id,
            rule_id=rule_id,
            care_type=payload.care_type,
            scheduled_at=payload.scheduled_at,
            status=payload.status,
        )
        return {"success": True, **res}
    except KeyError:
        raise HTTPException(status_code=404, detail="Care rule or item not found")


@router.delete("/{item_id}/care-rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_care_rule(item_id: str, rule_id: str, req: Request, user_id: Optional[str] = None):
    uid = enforce_owner(req, user_id)
    try:
        _care_service.delete_care_rule(user_id=uid, item_id=item_id, rule_id=rule_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except KeyError:
        raise HTTPException(status_code=404, detail="Care rule or item not found")


# ---------------------------------------------------------------------------
# 5. Soft Delete Endpoint (Idempotent 204)
# ---------------------------------------------------------------------------
@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_wardrobe_item(item_id: str, req: Request, user_id: Optional[str] = None):
    uid = enforce_owner(req, user_id)
    try:
        _item_service.delete_item(user_id=uid, item_id=item_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except KeyError:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
