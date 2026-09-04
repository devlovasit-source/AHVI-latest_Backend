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
    "and", "or", "but", "because", "with", "without", "for", "to", "like",
    "including", "while", "so", "of", "in", "on", "as", "if", "than", "then",
}

# Grammatical class: a sentence whose final word is a bare contracted
# subject+auxiliary/copula (e.g. "...so you're.") is missing its complement,
# the same incompleteness as _HANGING_ENDINGS' bare prepositions/conjunctions
# -- just a different part of speech. Checked via the same end-anchored
# last-word extraction below, so "You're ready for the meeting." (contraction
# mid-sentence) and "...you're after." (complement present, different final
# word) both stay valid; only the bare contraction as the sentence's very
# last token trips it.
_HANGING_CONTRACTED_AUX = {
    "i'm",
    "you're", "we're", "they're",
    "i've", "you've", "we've", "they've",
    "i'll", "you'll", "we'll", "they'll",
}

# Grammatical class: a sentence whose final word is a bare terminal article
# ("a"/"an"/"the") is missing the noun it introduces -- an article can never
# legitimately be the last word of a complete sentence. End-anchored via the
# same last-word check, so "the outfit" (article followed by its noun) never
# trips this; only the article as the sentence's very last token does.
_HANGING_BARE_ARTICLES = {"a", "an", "the"}

# Phrase-level detectors for constructions that end in valid terminal
# punctuation and a word outside _HANGING_ENDINGS, but are still
# grammatically incomplete because the phrasal verb is missing its
# object/complement (e.g. "...might feel out." — feel out OF WHAT?).
# Deliberately narrow: "stands out.", "head out.", "worked out.", "lights
# are out.", "feel off." must all keep validating as complete, so this
# matches only "feel(s/ing) out" as the sentence's final words, not any
# sentence merely containing or ending in "out".
#
# Unfinished trailing adjunct: a subordinator (without/while/before/after)
# followed by a gerund that's missing its object/complement. Two narrow,
# distinct shapes -- NOT a broad "any gerund clause is invalid" regex,
# which would wrongly flag genuinely complete bare-intransitive adjuncts
# like "without looking." / "without stopping." (existing valid cases):
#   1. subordinator + a SPECIFIC bare gerund ("being"/"maintaining") with
#      nothing after it (e.g. "...while maintaining."). Limited to the
#      gerunds proven to fail live, same precedent as "being" already was
#      -- not every transitive gerund, so "without looking."/"without
#      stopping." (intransitive, complete without an object) stay valid.
#   2. subordinator + ANY gerund + a bare trailing article (e.g. "...while
#      maintaining a." / "...while keeping the."). Safe to generalize to
#      any verb here: "<gerund> + a/an/the" as a sentence's last words is
#      never grammatically complete regardless of which verb it is.
# Both require real text after the gerund to stay valid: "while maintaining
# a polished silhouette.", "while keeping the palette neutral.", "without
# looking overly formal.", "after adding a lightweight layer." all keep
# validating as complete.
# ponytail: shape 1's whitelist only covers "being"/"maintaining" -- a new
# bare transitive gerund (e.g. "while balancing.") would slip through until
# added here; extend the alternation if that shape shows up live.
_TRAILING_ADJUNCT_RE = re.compile(
    r"\s*\b(?:without|while|before|after)\s+(?:being|maintaining)[.!?]*$"
    r"|\s*\b(?:without|while|before|after)\s+\w+ing\s+(?:a|an|the)[.!?]*$",
    re.IGNORECASE,
)

# Comma + bare terminal gerund/modifier with no complement (e.g. "...modern
# and refined, avoiding." -- avoiding WHAT?). Narrow: the gerund must be the
# ENTIRE remainder after the last comma, nothing else before the terminal
# punctuation -- a real participial modifier with its object stays valid:
# "...refined, avoiding loud contrasts." does not match since more text
# follows "avoiding" before the sentence ends.
# ponytail: a genuinely complete manner-participle with no object (e.g. "She
# left, smiling.") would also trip this -- same heuristic-vs-parser tradeoff
# as the trailing-adjunct check above; no valid-negative of that shape has
# been proven live, so it isn't special-cased.
_COMMA_GERUND_RE = re.compile(r",\s*\w+ing[.!?]*$", re.IGNORECASE)

_HANGING_PHRASE_PATTERNS = (
    re.compile(r"\bfeel(?:s|ing)?\s+out[.!?]*$", re.IGNORECASE),
    _TRAILING_ADJUNCT_RE,
    _COMMA_GERUND_RE,
)

_TERMINAL_MODIFIER_PATTERNS = (_TRAILING_ADJUNCT_RE, _COMMA_GERUND_RE)


def salvage_before_trailing_adjunct(text: str) -> str | None:
    """If `text` ends in a bare/underspecified terminal modifier -- a
    subordinate adjunct missing its complement ("...while maintaining a.")
    or a comma-delimited gerund missing its object ("...refined, avoiding.")
    -- strip that clause and return the leading sentence, but only when what
    remains is itself grammatically complete. Never invents the missing
    complement; returns None when there's nothing safe to salvage.
    """
    if not isinstance(text, str):
        return None
    original = text.strip()
    for pattern in _TERMINAL_MODIFIER_PATTERNS:
        salvaged = pattern.sub("", original).rstrip()
        salvaged = salvaged.rstrip(",;:-–— ")
        if not salvaged or salvaged == original:
            continue
        if not _TERMINAL_PUNCT_RE.search(salvaged):
            salvaged += "."
        if not looks_truncated(salvaged):
            return salvaged
    return None

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
        # A short label/noun ending in a single capitalized letter ("Plan
        # A.", "Vitamin A.") is a complete short sentence, not a truncated
        # one -- the length heuristic alone can't tell those apart, so give
        # it the same case-sensitive single-letter exemption the bare-article
        # check below relies on.
        short_last_word = re.split(r"\s+", stripped)[-1].strip(".,;:!?'\"()[]")
        if not (len(short_last_word) == 1 and short_last_word.isupper()):
            return True

    # Trailing connector punctuation.
    if stripped[-1] in {",", ":", ";", "-", "–", "—"}:
        return True

    # Hanging connector words / bare contracted auxiliaries missing a
    # complement.
    last_word_raw = re.split(r"\s+", stripped)[-1].strip(".,;:!?'\"()[]")
    last_word = last_word_raw.lower()
    if last_word in _HANGING_ENDINGS or last_word in _HANGING_CONTRACTED_AUX:
        return True

    # Bare terminal article missing its noun -- case-sensitive on purpose:
    # the function word "a"/"an"/"the" is essentially never capitalized
    # mid-sentence, so lowercasing here would misclassify a capitalized
    # single-letter label/noun ("Plan A.", "Vitamin A.") as the article.
    if last_word_raw in _HANGING_BARE_ARTICLES:
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
