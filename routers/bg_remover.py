"""
Backward-compatible background removal module.

Historically parts of the codebase referenced `routers.bg_remover`.
The canonical router implementation lives in `routers.bg_router.py`.
This module keeps those older imports working without forcing a rename.
"""

from __future__ import annotations

import base64
from typing import Any, Dict

from services.bg_service import remove_bg_external_sync


def remove_background_sync(image_base64: str) -> Dict[str, Any]:
    """
    Synchronous helper used by Celery tasks.
    Accepts base64 input and returns base64 output.
    """
    raw = str(image_base64 or "").strip()
    if not raw:
        return {"success": False, "error": "missing_image_base64"}
    image_bytes = base64.b64decode(raw.split(",")[-1])
    out_bytes = remove_bg_external_sync(image_bytes)
    return {"success": True, "image_base64": base64.b64encode(out_bytes).decode()}


def get_bg_runtime_metrics() -> Dict[str, Any]:
    """
    Lightweight metrics hook for /api/ops/metrics.
    Avoids importing the full router to keep dependency surface small.
    """
    return {"available": True, "backend": "hf+redis-cache", "mode": "sync_helper"}

