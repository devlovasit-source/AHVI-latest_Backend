"""Shared helper protecting embedded newlines through text post-processors
that flatten whitespace (e.g. brain.tone.tone_engine.apply).

Pure module: stdlib only. Safe to import from any layer (services, routers)
without pulling in tone_engine or LLM dependencies, so both llm_service.py
and style_reasoning_engine.py can share one implementation instead of two
copies drifting apart.
"""

from __future__ import annotations

from typing import Callable

# U+2063 INVISIBLE SEPARATOR -- not whitespace, so it survives str.split()
# and every \s-based regex a text post-processor uses, unlike a placeholder
# built from ordinary characters.
_NEWLINE_SENTINEL = "⁣"


def protect_newlines_through(text: str, transform: Callable[[str], str]) -> str:
    """Run `text` through `transform` without losing embedded newlines.

    `transform` is any str->str callable that may collapse whitespace
    (tone_engine.apply is the motivating case). Newlines are encoded as an
    invisible sentinel before the call and decoded back after, so callers
    that need bullet/line-break structure preserved (e.g. STYLE_ADVICE_
    FORMAT_CONTRACT responses) survive tone processing intact.
    """
    protected = str(text or "").replace("\n", _NEWLINE_SENTINEL)
    result = transform(protected)
    return str(result or "").replace(_NEWLINE_SENTINEL, "\n")
