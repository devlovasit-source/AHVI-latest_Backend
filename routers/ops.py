import subprocess
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


def _gpu_metrics() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=2.0,
        )
        rows = []
        for line in (out or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            rows.append(
                {
                    "gpu_index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(float(parts[2])),
                    "memory_total_mb": int(float(parts[3])),
                    "utilization_percent": int(float(parts[4])),
                    "temperature_c": int(float(parts[5])),
                }
            )
        return {"available": True, "gpus": rows}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


@router.get("/metrics")
def ops_metrics():
    llm_metrics = {}
    bg_metrics = {}

    try:
        from services.llm_service import get_runtime_metrics

        llm_metrics = get_runtime_metrics()
    except Exception as exc:
        llm_metrics = {"error": str(exc)}

    try:
        from routers.bg_remover import get_bg_runtime_metrics

        bg_metrics = get_bg_runtime_metrics()
    except Exception as exc:
        bg_metrics = {"error": str(exc)}

    return {
        "success": True,
        "llm": llm_metrics,
        "background_queue": bg_metrics,
        "gpu": _gpu_metrics(),
    }


class VisionIntelligenceCheckRequest(BaseModel):
    image_base64: str | None = Field(default=None, min_length=20)
    userId: str = "ops_diagnostic_user"
    run_live_vision: bool = False


def _read_path(data: dict[str, Any], path: str) -> Any:
    node: Any = data
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node.get(key)
    return node


def _validate_vision_contract(payload: dict[str, Any]) -> dict[str, Any]:
    required_paths = [
        "success",
        "data.name",
        "data.category",
        "data.sub_category",
        "data.pattern",
        "data.color_code",
        "items",
        "outfit.score",
        "outfit.analysis.completeness",
        "style.tone",
        "style.color_tone",
        "visual_intelligence.dominant_color_hex",
        "visual_intelligence.temperature_tone",
        "visual_intelligence.expression_tone",
        "meta.tone_engine_used",
        "meta.visual_intelligence_enabled",
    ]
    missing_fields = [path for path in required_paths if _read_path(payload, path) is None]

    invalid_fields = []
    style_tone = _read_path(payload, "style.tone")
    if style_tone not in {"minimal", "expressive"}:
        invalid_fields.append("style.tone")

    color_tone = _read_path(payload, "style.color_tone")
    if color_tone not in {"warm", "cool", "neutral"}:
        invalid_fields.append("style.color_tone")

    temp_tone = _read_path(payload, "visual_intelligence.temperature_tone")
    if temp_tone not in {"warm", "cool", "neutral"}:
        invalid_fields.append("visual_intelligence.temperature_tone")

    expression_tone = _read_path(payload, "visual_intelligence.expression_tone")
    if expression_tone not in {"minimal", "expressive"}:
        invalid_fields.append("visual_intelligence.expression_tone")

    return {
        "contract_ok": not missing_fields and not invalid_fields,
        "missing_fields": missing_fields,
        "invalid_fields": invalid_fields,
        "required_paths_checked": required_paths,
    }


@router.post("/vision-intelligence-check")
def vision_intelligence_check(request: VisionIntelligenceCheckRequest):
    synthetic_payload: dict[str, Any] = {
        "success": True,
        "data": {
            "name": "Black Shirt",
            "category": "Tops",
            "sub_category": "Shirt",
            "pattern": "plain",
            "color_code": "#111111",
            "userId": request.userId,
        },
        "items": [{"type": "shirt", "color": "#111111", "style": "casual"}],
        "outfit": {"score": 80, "analysis": {"completeness": "partial"}},
        "style": {"tone": "minimal", "color_tone": "neutral", "versatility": "high"},
        "visual_intelligence": {
            "dominant_color_hex": "#111111",
            "dominant_color_name": "Black",
            "temperature_tone": "neutral",
            "expression_tone": "minimal",
            "item_type": "Shirt",
            "style_consistency": "cohesive",
            "color_harmony": "clean",
            "completeness": "partial",
            "items_detected": 1,
        },
        "meta": {
            "tone_engine_used": True,
            "visual_intelligence_enabled": True,
        },
    }

    used_live_vision = bool(request.run_live_vision and request.image_base64)
    payload = synthetic_payload
    live_error = None

    if used_live_vision:
        try:
            from routers.vision import vision_analyze_core

            payload = vision_analyze_core(request.image_base64 or "", request.userId)
        except Exception as exc:
            live_error = str(exc)
            payload = synthetic_payload
            used_live_vision = False

    checks = _validate_vision_contract(payload)
    return {
        "success": True,
        "checks": checks,
        "used_live_vision": used_live_vision,
        "live_vision_error": live_error,
        "snapshot": {
            "style": _read_path(payload, "style") or {},
            "visual_intelligence": _read_path(payload, "visual_intelligence") or {},
            "meta": _read_path(payload, "meta") or {},
        },
    }
