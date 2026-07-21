"""Job-only, write-denied regression runner for the Style candidate.

This module is deliberately not a FastAPI router.  It is imported only by an
operator job, uses a configured dedicated identity, and has no caller supplied
prompt, identity, policy, or budget surface.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from routers import stylist
from services.appwrite_proxy import AppwriteProxy
from services.style_execution_policy import (
    activate_style_execution,
    create_style_execution_session,
)


SCENARIO_VERSION = "batch11p-v1"

# Only stable server-side identifiers are kept here.  The repository remains
# authoritative for every other field, including image, source and metadata.
_SCENARIOS = (
    ("very_hot_daily", "mens_assets_tops_blackshirt", "daily", {"temperature_c": 36, "condition": "very_hot"}),
    ("client_meeting", "mens_assets_tops_blackshirt", "client_meeting", {}),
    ("cultural_professional", "mens_assets_festive_sets_bluekurta", "office", {}),
    ("cold_accessory", "mens_assets_outerwear_blackplaidscarf", "daily", {"temperature_c": 8, "condition": "cold"}),
    ("unknown_weather", "meghna_female_top_black_and_white_stripe_tshirt", "daily", {}),
)


class CandidateEvaluationConfigurationError(RuntimeError):
    pass


class CandidateEvaluationWriteError(RuntimeError):
    pass


def _configured_test_identity() -> str:
    """Return only an explicitly enabled, exactly authorized job identity."""
    if str(os.getenv("STYLE_EVALUATION_RUNNER_ENABLED") or "").strip().lower() != "true":
        raise CandidateEvaluationConfigurationError("candidate evaluation runner is disabled")
    identity = str(os.getenv("STYLE_EVALUATION_TEST_USER_ID") or "").strip()
    authorized = str(os.getenv("STYLE_EVALUATION_AUTHORIZED_USER_ID") or "").strip()
    if not identity or not authorized or identity != authorized:
        raise CandidateEvaluationConfigurationError("candidate evaluation identity is not authorized")
    return identity


def _load_style_assets() -> List[Dict[str, Any]]:
    """The evaluator has an intentionally read-only Appwrite surface."""
    rows = AppwriteProxy().list_documents("style_assets") or []
    return [
        {**dict(row), "source": dict(row).get("source") or "style_asset"}
        for row in rows
        if isinstance(row, dict)
    ]


def _safe_result(name: str, response: Dict[str, Any]) -> Dict[str, Any]:
    directions = response.get("style_directions") if isinstance(response, dict) else []
    selected = []
    for direction in directions if isinstance(directions, list) else []:
        for item in direction.get("items") or []:
            if isinstance(item, dict):
                value = str(item.get("item_id") or item.get("id") or "").strip()
                if value and value not in selected:
                    selected.append(value)
    anchor = response.get("anchor_item") if isinstance(response, dict) else {}
    return {
        "scenario": name,
        "success": bool(response.get("success")) if isinstance(response, dict) else False,
        "anchor_id": str((anchor or {}).get("asset_id") or (anchor or {}).get("$id") or ""),
        "selected_asset_ids": selected,
        "anchor_blocked": bool(response.get("anchor_blocked")) if isinstance(response, dict) else True,
        "source": str(response.get("source") or "") if isinstance(response, dict) else "",
        "error_code": str(((response.get("error") or {}) if isinstance(response, dict) else {}).get("code") or ""),
    }


def run_internal_candidate_evaluation() -> Dict[str, Any]:
    """Run the fixed candidate suite under a server-created no-write policy.

    There are deliberately no parameters.  A deployment job must configure the
    dedicated identity before this function can read any data.
    """
    identity = _configured_test_identity()
    assets = _load_style_assets()
    session = create_style_execution_session("read_only_evaluation")
    results = []
    with activate_style_execution(session):
        for name, asset_id, occasion, weather in _SCENARIOS:
            request = stylist.ItemStyleRequest(
                user_id=identity,
                mode="style_this",
                occasion=occasion,
                wardrobe=[],
                # This value is loaded by this module from the server
                # repository, never passed by an HTTP caller.
                style_assets=assets,
                context={"weather_context": dict(weather)},
            )
            response = stylist.style_wardrobe_item(asset_id, request, None)
            results.append(_safe_result(name, response if isinstance(response, dict) else {}))
    if session.blocked_write_attempts:
        raise CandidateEvaluationWriteError("candidate evaluation attempted a denied write")
    return {
        "scenario_version": SCENARIO_VERSION,
        "scenario_count": len(results),
        "results": results,
        "model_calls": session.budget.count,
        "blocked_write_attempts": session.blocked_write_attempts,
        "image_generation_calls": 0,
    }


__all__ = [
    "CandidateEvaluationConfigurationError",
    "CandidateEvaluationWriteError",
    "SCENARIO_VERSION",
    "run_internal_candidate_evaluation",
]
