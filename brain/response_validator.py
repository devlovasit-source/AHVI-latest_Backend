import re
import json
import logging
from typing import Any, Dict, List

_CODE_FENCE_RE = re.compile(r"```(?:json|python|text)?|```", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")

_TERMINAL_PUNCT_RE = re.compile(r"[.!?…]['\")\]]?$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-Z\"'(\[])")
_HANGING_ENDINGS = {
    "and", "or", "but", "because", "with", "for", "to", "like", "while",
    "so", "of", "in", "on", "as", "if", "than", "then",
}

# Phrase-level detectors for constructions that end in valid terminal
# punctuation and a word outside _HANGING_ENDINGS, but are still
# grammatically incomplete because the phrasal verb is missing its
# object/complement (e.g. "...might feel out." — feel out OF WHAT?).
# Deliberately narrow: "stands out.", "head out.", "worked out.", "lights
# are out.", "feel off." must all keep validating as complete, so this
# matches only "feel(s/ing) out" as the sentence's final words, not any
# sentence merely containing or ending in "out".
#
# Unfinished copular gerund after a subordinator/preposition (e.g.
# "...without being." — without being WHAT?). Narrow on purpose: only the
# specific subordinators below, immediately followed by bare "being" as the
# sentence's final word. "Being prepared matters." (sentence-initial),
# "without looking."/"without stopping." (a different, complete gerund), and
# "without being overly formal." (has its complement) must all stay valid.
_HANGING_PHRASE_PATTERNS = (
    re.compile(r"\bfeel(?:s|ing)?\s+out[.!?]*$", re.IGNORECASE),
    re.compile(r"\b(?:without|while|before|after)\s+being[.!?]*$", re.IGNORECASE),
)

_FORBIDDEN_STARTERS = (
    "Sure!",
    "Absolutely!",
    "Great choice!",
    "Here are some ideas",
    "Okay wait",
    "Not gonna lie",
    "Would you like me to",
    "You ate that",
    "This eats",
)

logger = logging.getLogger("ahvi.response_validator")


def truncate_preserving_sentence(text: str, max_chars: int = 2000) -> str:
    """Truncate `text` to <= max_chars without cutting mid-sentence/word."""
    if not text or len(text) <= max_chars:
        return text
    window = text[:max_chars]
    # Prefer the last terminal punctuation inside the window.
    last_terminal = max(window.rfind("."), window.rfind("!"), window.rfind("?"), window.rfind("…"))
    if last_terminal >= int(max_chars * 0.5):
        return window[: last_terminal + 1].rstrip()
    # Fall back to last whitespace so we never cut mid-word.
    last_space = window.rfind(" ")
    if last_space >= int(max_chars * 0.4):
        return window[:last_space].rstrip() + "…"
    return window.rstrip() + "…"


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
    text = truncate_preserving_sentence(text, max_chars=2000)
    return text


def _balanced(text: str, open_ch: str, close_ch: str) -> bool:
    return text.count(open_ch) == text.count(close_ch)


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped[0] in "{[" and stripped[-1] in "}]"


def looks_truncated(text: str) -> bool:
    """Heuristic check: does this look cut off mid-thought?"""
    if not isinstance(text, str):
        return True
    stripped = text.strip()
    if len(stripped) < 8:
        return True

    # Trailing connector punctuation.
    if stripped[-1] in {",", ":", ";", "-", "–", "—"}:
        return True

    # Hanging connector words.
    last_word = re.split(r"\s+", stripped)[-1].strip(".,;:!?'\"()[]").lower()
    if last_word in _HANGING_ENDINGS:
        return True

    # Hanging phrasal constructions (e.g. "...might feel out.") that end in
    # valid terminal punctuation and a word outside _HANGING_ENDINGS but are
    # still missing a required object/complement.
    if any(pattern.search(stripped) for pattern in _HANGING_PHRASE_PATTERNS):
        return True

    # Unclosed code fence.
    if stripped.count("```") % 2 == 1:
        return True

    # Unbalanced brackets / quotes.
    for op, cl in (("(", ")"), ("[", "]"), ("{", "}")):
        if not _balanced(stripped, op, cl):
            return True
    if stripped.count('"') % 2 == 1:
        return True

    # Bullet line ending abruptly.
    lines = [ln.rstrip() for ln in stripped.split("\n") if ln.strip()]
    if lines:
        last_line = lines[-1]
        if last_line.startswith(("- ", "* ", "• ")) and not _TERMINAL_PUNCT_RE.search(last_line):
            last_word2 = re.split(r"\s+", last_line)[-1].strip(".,;:!?'\"()[]").lower()
            if last_word2 in _HANGING_ENDINGS or len(last_line) < 12:
                return True

    # JSON-looking output that does not actually parse.
    if _looks_like_json(stripped):
        try:
            json.loads(stripped)
            return False
        except Exception:
            return True

    # Missing terminal punctuation for normal prose.
    if not _TERMINAL_PUNCT_RE.search(stripped):
        return True

    return False


def validate_final_text(
    text: str,
    *,
    fallback: str = "I can make this cleaner and more specific.",
) -> Dict[str, Any]:
    issues: List[str] = []
    raw = str(text or "")
    polished = to_plain_text(raw, fallback=fallback)

    truncated = looks_truncated(polished)
    if truncated:
        issues.append("looks_truncated")

    if not polished or polished == fallback:
        issues.append("empty_or_fallback")

    return {
        "text": polished,
        "is_valid": not issues,
        "looks_truncated": truncated,
        "issues": issues,
    }


def polish_final_text(
    text: str,
    *,
    fallback: str = "I can make this cleaner and more specific.",
) -> str:
    """Final text gate: strip forbidden starters, repair truncation, fall
    back when nothing salvageable remains."""
    cleaned = to_plain_text(text, fallback=fallback)
    for starter in _FORBIDDEN_STARTERS:
        # Case-insensitive match anchored at start, allowing optional space.
        pat = re.compile(r"^\s*" + re.escape(starter) + r"\s*[,.!?:-]*\s*", re.IGNORECASE)
        if pat.search(cleaned):
            cleaned = pat.sub("", cleaned, count=1).strip()
            try:
                logger.info("ahvi.response.polished removed_starter=%r", starter)
            except Exception:
                pass

    if not cleaned or len(cleaned) < 3:
        return fallback

    if looks_truncated(cleaned):
        # Trim to the last complete sentence.
        sentences = _SENTENCE_SPLIT_RE.split(cleaned)
        complete = [s for s in sentences if _TERMINAL_PUNCT_RE.search(s.strip())]
        if complete:
            trimmed = " ".join(s.strip() for s in complete).strip()
            try:
                logger.info("ahvi.response.truncated_detected action=trimmed")
            except Exception:
                pass
            if trimmed and not looks_truncated(trimmed):
                return trimmed
        try:
            logger.warning("ahvi.response.truncated_detected action=fallback")
        except Exception:
            pass
        return fallback

    return cleaned


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
