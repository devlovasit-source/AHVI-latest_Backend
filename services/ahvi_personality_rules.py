import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("ahvi.personality_rules")

_P0_FILES: tuple[str, ...] = (
    "persona_normalized.json",
    "tone_rules_normalized.json",
    "behavior_rules_normalized.json",
    "response_priorities_normalized.json",
    "decision_frameworks_normalized.json",
    "visual_rules_normalized.json",
)

_NORMALIZED_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "ahvi_personality_normalized"
)


@lru_cache(maxsize=1)
def load_personality_rules() -> Dict[str, Any]:
    """Load normalized P0 personality/tone rules.

    This is deliberately tiny and fail-open. Styling must never block because
    a copy deck is absent or malformed.
    """
    loaded: Dict[str, Any] = {}
    failed: Dict[str, str] = {}

    for filename in _P0_FILES:
        path = _NORMALIZED_DIR / filename
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded[filename] = json.load(handle)
        except Exception as exc:  # noqa: BLE001 - fail open by design.
            failed[filename] = str(exc)[:240]
            logger.warning(
                "ahvi.personality_rules.failed file=%s error=%s",
                filename,
                str(exc)[:240],
            )

    logger.info(
        "ahvi.personality_rules.loaded loaded=%d failed=%d source=%s",
        len(loaded),
        len(failed),
        str(_NORMALIZED_DIR),
    )
    return {
        "loaded": loaded,
        "failed": failed,
        "source_dir": str(_NORMALIZED_DIR),
    }


__all__ = ["load_personality_rules"]
