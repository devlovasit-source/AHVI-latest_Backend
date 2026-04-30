import re
import logging
from typing import Any, Dict


_CODE_FENCE_RE = re.compile(r"```(?:json|python|text)?|```", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")

logger = logging.getLogger("ahvi.response_validator")


def to_plain_text(value: Any, *, fallback: str = "I can help with that.") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback

    text = _CODE_FENCE_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r", "\n")
    text = _MULTISPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    text = text.strip()
    if not text:
        return fallback
    if len(text) > 2000:
        text = text[:2000].rstrip() + "..."
    return text


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for x in value:
            s = str(x or "").strip()
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        return [p for p in parts if p]
    return []


def _first_id(value: Any) -> str:
    if isinstance(value, str):
        return str(value).strip()
    if isinstance(value, list) and value:
        return str(value[0]).strip()
    return ""


def _sanitize_cards(value: Any) -> list[dict]:
    """
    UI-safe card normalization:
    - keep only dict cards
    - ensure required keys exist (id/title/items)
    - coerce items to a list
    """
    cards_in = value if isinstance(value, list) else []
    out: list[dict] = []
    for idx, raw in enumerate(cards_in):
        if not isinstance(raw, dict):
            continue
        card = dict(raw)
        card_id = str(card.get("id") or f"card_{idx + 1}").strip() or f"card_{idx + 1}"
        title = to_plain_text(card.get("title"), fallback="Outfit")
        items = card.get("items")
        if not isinstance(items, list):
            items = []
        card["id"] = card_id
        card["title"] = title
        card["items"] = items
        if "score" in card:
            try:
                card["score"] = float(card.get("score") or 0.0)
            except Exception:
                card["score"] = 0.0
        out.append(card)
    return out


def validate_orchestrator_response(
    payload: Dict[str, Any] | Any,
    *,
    request_id: str = "",
) -> Dict[str, Any]:
    row = dict(payload) if isinstance(payload, dict) else {}

    # Required top-level safety defaults.
    row["success"] = bool(row.get("success", True))
    row["request_id"] = str(row.get("request_id") or request_id or "")
    row["message"] = to_plain_text(
        row.get("message"),
        fallback="I can help with that.",
    )

    row["cards"] = _sanitize_cards(row.get("cards", []))
    data = row.get("data", {})
    row["data"] = data if isinstance(data, dict) else {}
    meta = row.get("meta", {})
    row["meta"] = meta if isinstance(meta, dict) else {}
    row["board"] = str(row.get("board") or "general")
    row["type"] = str(row.get("type") or "text")

    # Contract hardening: board_ids/pack_ids are consumed by the Flutter client as a single id string.
    board_ids = row.get("board_ids", row.get("board_id", ""))
    pack_ids = row.get("pack_ids", row.get("pack_id", ""))
    row["board_ids"] = _first_id(board_ids)
    row["pack_ids"] = _first_id(pack_ids)

    # If the engine produced a list of ids, preserve it in data for future UIs without breaking old clients.
    if "board_item_ids" not in row["data"]:
        ids = _as_str_list(board_ids)
        if ids:
            row["data"]["board_item_ids"] = ids

    try:
        logger.info(
            "validated request_id=%s type=%s board=%s cards=%s board_ids=%s",
            row.get("request_id"),
            row.get("type"),
            row.get("board"),
            len(row.get("cards") or []),
            row.get("board_ids") or "",
        )
    except Exception:
        pass

    return row

