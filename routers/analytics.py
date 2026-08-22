"""Canonical Clickstream Analytics Router.

Exposes REST APIs for clickstream event batch ingestion and analytical queries
(DAU, Feature Usage, Funnel Drop-offs, Sessionized User Journeys).
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from services.clickstream_service import ClickstreamService

logger = logging.getLogger("ahvi.routers.analytics")

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

_clickstream = ClickstreamService()


class EventPayload(BaseModel):
    event_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    event_name: str
    timestamp: str
    screen: Optional[str] = None
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)
    device: Optional[Dict[str, Any]] = Field(default_factory=dict)
    app_version: Optional[str] = "1.0.0"


class IngestBatchRequest(BaseModel):
    events: List[EventPayload]


def _check_admin_auth(x_admin_key: Optional[str] = None) -> None:
    expected_key = os.getenv("ANALYTICS_ADMIN_KEY", "").strip()
    if expected_key:
        provided = (x_admin_key or "").strip()
        if provided != expected_key:
            raise HTTPException(status_code=403, detail="Invalid admin credentials")


@router.post("/events")
def batch_ingest_events(payload: IngestBatchRequest):
    try:
        raw_list = [e.dict() for e in payload.events]
        res = _clickstream.ingest_events(raw_list)
        return {"success": True, **res}
    except Exception as exc:
        logger.warning("ahvi.analytics.ingest_failed err=%s", str(exc)[:140])
        return {"success": False, "accepted": 0, "error": str(exc)}


@router.get("/metrics")
def get_analytics_metrics(x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")):
    _check_admin_auth(x_admin_key)
    try:
        metrics = _clickstream.get_metrics()
        return {"success": True, **metrics}
    except Exception as exc:
        logger.warning("ahvi.analytics.metrics_failed err=%s", str(exc)[:140])
        raise HTTPException(status_code=500, detail="Failed to compute analytics metrics")


@router.get("/funnel")
def get_funnel_analytics(x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")):
    _check_admin_auth(x_admin_key)
    try:
        funnel_data = _clickstream.get_funnel_analysis()
        return {"success": True, **funnel_data}
    except Exception as exc:
        logger.warning("ahvi.analytics.funnel_failed err=%s", str(exc)[:140])
        raise HTTPException(status_code=500, detail="Failed to compute funnel analytics")


@router.get("/user-journeys")
def get_user_journeys(
    limit: int = 20,
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
):
    _check_admin_auth(x_admin_key)
    try:
        journeys = _clickstream.get_user_journeys(limit_sessions=limit)
        return {"success": True, "user_journeys": journeys}
    except Exception as exc:
        logger.warning("ahvi.analytics.user_journeys_failed err=%s", str(exc)[:140])
        raise HTTPException(status_code=500, detail="Failed to compute user journeys")
