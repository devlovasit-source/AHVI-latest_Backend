"""FastAPI Router for AHVI Temporal Intelligence endpoints.

Exposes unified timeline streams, temporal context queries, opportunity registry,
live shadow mode evaluation, and attention arbitration API endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from brain.temporal.aggregation_service import aggregation_service
from brain.temporal.attention_arbitrator import attention_arbitrator
from brain.temporal.attention_models import CandidateAction
from brain.temporal.context_engine import context_engine
from brain.temporal.opportunity_store import opportunity_store
from brain.temporal.shadow_mode import shadow_evaluator

logger = logging.getLogger("ahvi.routers.temporal")

router = APIRouter(prefix="/api/temporal", tags=["Temporal Intelligence"])


class ShadowModeEvalRequest(BaseModel):
    user_id: str = Field(default="shadow_user", description="Target user ID")
    raw_events: List[Dict[str, Any]] = Field(default_factory=list, description="Raw calendar events list")
    triaged_diff_keys: Optional[List[str]] = Field(default=None, description="Optional list of triaged diff keys")


class ArbitrationRequest(BaseModel):
    user_id: str = Field(default="user_1", description="Target user ID")
    candidate_actions: List[CandidateAction] = Field(default_factory=list, description="List of candidate actions to arbitrate")


@router.get("/timeline")
def get_unified_timeline(
    request: Request,
    user_id: Optional[str] = Query(default=None, description="Target user ID"),
) -> Dict[str, Any]:
    """Fetch the unified, aggregated timeline stream across all module adapters."""
    uid = user_id or getattr(request.state, "user", {}).get("id") or "default_user"
    items = aggregation_service.fetch_unified_timeline(uid)
    return {
        "success": True,
        "user_id": uid,
        "count": len(items),
        "timeline_items": [it.model_dump() for it in items],
    }


@router.get("/context")
def get_temporal_context(
    request: Request,
    user_id: Optional[str] = Query(default=None, description="Target user ID"),
    window_hours: float = Query(default=24.0, description="Lookahead window in hours"),
) -> Dict[str, Any]:
    """Query current activity and upcoming activities from Temporal Context Engine."""
    uid = user_id or getattr(request.state, "user", {}).get("id") or "default_user"
    timeline_items = aggregation_service.fetch_unified_timeline(uid)

    current_act = context_engine.get_current_activity(uid, items=timeline_items)
    upcoming = context_engine.get_upcoming_activities(uid, window_hours=window_hours, items=timeline_items)
    conflicts = context_engine.detect_schedule_conflicts(timeline_items)

    return {
        "success": True,
        "user_id": uid,
        "current_activity": current_act.model_dump() if current_act else None,
        "upcoming_count": len(upcoming),
        "upcoming_activities": [it.model_dump() for it in upcoming],
        "schedule_conflicts": conflicts,
    }


@router.get("/opportunities")
def get_user_opportunities(
    request: Request,
    user_id: Optional[str] = Query(default=None, description="Target user ID"),
) -> Dict[str, Any]:
    """Query persisted opportunities for a user from OpportunityStore."""
    uid = user_id or getattr(request.state, "user", {}).get("id") or "default_user"
    opps = opportunity_store.query_user_opportunities(uid)
    return {
        "success": True,
        "user_id": uid,
        "count": len(opps),
        "opportunities": [opp.model_dump() for opp in opps],
    }


@router.post("/evaluate-shadow-mode")
def evaluate_shadow_mode(payload: ShadowModeEvalRequest) -> Dict[str, Any]:
    """Execute side-by-side comparison between legacy Calendar Intelligence and Temporal Context Engine."""
    res = shadow_evaluator.evaluate_events(
        raw_events=payload.raw_events,
        user_id=payload.user_id,
        triaged_diff_keys=payload.triaged_diff_keys,
    )
    return {
        "success": True,
        "evaluation": res,
    }


@router.post("/arbitrate")
def arbitrate_attention(payload: ArbitrationRequest) -> Dict[str, Any]:
    """Arbitrate competing candidate actions to prevent user notification fatigue."""
    arbitrated = attention_arbitrator.arbitrate_actions(payload.candidate_actions)
    return {
        "success": True,
        "user_id": payload.user_id,
        "input_count": len(payload.candidate_actions),
        "output_count": len(arbitrated),
        "arbitrated_actions": [act.model_dump() for act in arbitrated],
    }
