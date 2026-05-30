"""Vertex AI Reasoning Engine transport for AHVI agent calls.

Both `agent_style_orchestrator` and `agent_metadata_validator` started
shipping with `AGENT_*_ENDPOINT` set to a Vertex Reasoning Engine
*resource id* (e.g. `projects/631493992863/locations/us-west1/reasoningEngines/6180706703449784320`)
rather than a real `https://...` URL. The old httpx path on those
strings raised:

    Request URL is missing an 'http://' or 'https://' protocol.

This helper detects that pattern and calls the Reasoning Engine via the
official Vertex AI Python SDK. Never raises — returns parsed dict or
None so callers fall back to legacy logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ahvi.agent.reasoning_engine")


_RESOURCE_RE = re.compile(
    r"^projects/[^/]+/locations/([^/]+)/reasoningEngines/[^/]+$"
)


def looks_like_resource_id(endpoint: Any) -> bool:
    """True if `endpoint` is a Vertex Reasoning Engine resource id."""
    if not isinstance(endpoint, str):
        return False
    text = endpoint.strip()
    if not text.startswith("projects/"):
        return False
    return "/reasoningEngines/" in text


def _location_from_resource(resource_id: str, fallback: str = "us-central1") -> str:
    m = _RESOURCE_RE.match(resource_id.strip())
    return m.group(1) if m else fallback


def _project_from_env() -> str:
    return (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or os.getenv("AHVI_GCP_PROJECT")
        or ""
    )


def _parse_engine_payload(payload: Any) -> Optional[Dict[str, Any]]:
    """Normalize the various Reasoning Engine / ADK response envelopes
    into a flat dict ready for the agent schema validator."""
    if payload is None:
        return None
    if isinstance(payload, dict):
        # ADK-style: {"output": {"content": "..."}} or {"output": {"text": "..."}}
        output = payload.get("output")
        if isinstance(output, dict):
            content = output.get("content") or output.get("text")
            if isinstance(content, str):
                try:
                    return json.loads(content)
                except Exception:
                    return None
            if isinstance(content, dict):
                return content
        # Plain content string at the top level.
        if isinstance(payload.get("content"), str):
            try:
                return json.loads(payload["content"])
            except Exception:
                return None
        # Already a flat dict — return as-is. The schema validator will
        # coerce missing keys.
        return payload
    if isinstance(payload, str):
        text = payload.strip()
        # ADK / LLM models sometimes wrap JSON in markdown code fences
        # despite the strict-output instructions.
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except Exception:
            # Last-ditch: pull the first {...} block.
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None
    return None


def _load_engine(resource_id: str):
    """Return a live engine handle from the Vertex SDK.

    Newer Vertex SDKs expose Agent Engines via `vertexai.agent_engines.get`.
    Older / preview ones use `vertexai.preview.reasoning_engines.ReasoningEngine`.
    Try both so we keep working across SDK versions.
    """
    # Newer API surface (preferred).
    try:
        from vertexai import agent_engines  # type: ignore
        return agent_engines.get(resource_id.strip())
    except Exception:
        pass
    # Preview API surface.
    try:
        from vertexai.preview import reasoning_engines  # type: ignore
        # Some SDK builds support a `get(resource_id)` constructor; others
        # accept the resource id directly on the class.
        if hasattr(reasoning_engines, "ReasoningEngine"):
            return reasoning_engines.ReasoningEngine(resource_id.strip())
    except Exception:
        pass
    return None


_METHOD_NAMES = (
    "query",
    "stream_query",
    "stream_query_text",
    "run",
    "run_live",
    "invoke",
    "predict",
    "generate",
    "chat",
    "send",
    "complete",
    "respond",
    "__call__",
)


def _list_engine_callables(engine) -> List[str]:
    """Return every public callable attribute on the engine handle.

    Used in warning logs so we can see — without re-deploying — what
    method names the deployed Agent Engine actually exposes (these are
    bound dynamically per agent).
    """
    out: List[str] = []
    try:
        for name in dir(engine):
            if name.startswith("_"):
                continue
            try:
                attr = getattr(engine, name)
            except Exception:
                continue
            if callable(attr):
                out.append(name)
    except Exception:
        pass
    return sorted(out)


def _extract_text_from_adk_events(events: List[Any]) -> str:
    """Pull the final assistant text out of an ADK stream_query event list.

    ADK emits events like:
      {"content": {"parts": [{"text": "..."}]}, "author": "agent", ...}
    The system prompt + reasoning trail also flow through; the assistant's
    final answer is the last event whose `content.parts[*].text` is non-empty.
    """
    last_text = ""
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        content = ev.get("content")
        if not isinstance(content, dict):
            # Some events use {"output": "..."} or plain text.
            txt = ev.get("output") or ev.get("text")
            if isinstance(txt, str) and txt.strip():
                last_text = txt
            continue
        parts = content.get("parts")
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, dict):
                    txt = p.get("text")
                    if isinstance(txt, str) and txt.strip():
                        last_text = txt
    return last_text


def _ensure_session(engine, user_id: str) -> Optional[str]:
    """ADK Agent Engines expect a session id on every stream_query call.

    Try the documented surfaces; return None when the engine has no
    session concept (older Reasoning Engines that accept naked input).
    """
    for fn_name in ("create_session", "start_session"):
        fn = getattr(engine, fn_name, None)
        if not callable(fn):
            continue
        try:
            sess = fn(user_id=user_id)
        except TypeError:
            try:
                sess = fn(user_id)
            except Exception:
                continue
        except Exception:
            continue
        if isinstance(sess, dict):
            return str(sess.get("id") or sess.get("session_id") or "")
        sid = getattr(sess, "id", None) or getattr(sess, "session_id", None)
        if sid:
            return str(sid)
    return None


def _call_engine_with_fallbacks(engine, system: str, prompt: str):
    """Try every plausible method + signature on the engine handle.

    Prefer ADK signatures (stream_query with message/user_id/session_id)
    since that's what AHVI's Style Orchestrator and Metadata Validator
    deploy as. The system prompt is baked into the agent at build time
    so we only send the user prompt here.
    """
    user_id = "ahvi-backend"
    session_id = _ensure_session(engine, user_id)

    # ADK Agent Engine call patterns (message-based + session-scoped).
    adk_payloads: List[Dict[str, Any]] = [
        {"message": prompt, "user_id": user_id, "session_id": session_id or ""},
        {"message": prompt, "user_id": user_id},
        {"message": prompt},
    ]
    # Generic Reasoning Engine fallback payloads.
    generic_payloads: List[Dict[str, Any]] = [
        {"input": {"system": system, "prompt": prompt}},
        {"input": f"{system}\n\n{prompt}"},
        {"input": prompt},
        {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]},
    ]

    last_exc: Optional[Exception] = None
    for name in _METHOD_NAMES:
        fn = getattr(engine, name, None)
        if not callable(fn):
            continue
        is_stream = name.startswith("stream")
        # ADK signatures first for stream_query, generic first for query/run.
        payloads = (adk_payloads + generic_payloads) if is_stream else (generic_payloads + adk_payloads)
        for payload in payloads:
            # Strip session_id when blank to avoid TypeError on strict engines.
            clean_payload = {k: v for k, v in payload.items() if v != ""}
            try:
                result = fn(**clean_payload)
            except TypeError:
                try:
                    first_val = next(iter(clean_payload.values()))
                    result = fn(first_val)
                except Exception as exc:
                    last_exc = exc
                    continue
            except Exception as exc:
                last_exc = exc
                continue

            # Drain generators (stream_query yields events).
            if hasattr(result, "__iter__") and not isinstance(result, (dict, str, bytes, list)):
                try:
                    events = list(result)
                except Exception as exc:
                    last_exc = exc
                    continue
                if not events:
                    last_exc = RuntimeError(f"{name} returned empty stream")
                    continue
                # If it looks like an ADK event list, pull the assistant text.
                text = _extract_text_from_adk_events(events)
                if text:
                    return text, name
                # Otherwise return the last chunk as-is.
                return events[-1], name
            if isinstance(result, list):
                text = _extract_text_from_adk_events(result)
                if text:
                    return text, name
                return result, name
            return result, name
    if last_exc is not None:
        raise last_exc
    raise AttributeError(
        f"engine exposes no callable in {list(_METHOD_NAMES)}"
    )


def _discover_operation_names(resource_id: str) -> List[str]:
    """Return the class_method names registered on the deployed agent.

    Uses `agent.operation_schemas()` (available on AgentEngine handles)
    to ask Vertex which operations were registered when the agent was
    deployed. Falls back to ('stream_query', 'query') if discovery
    fails for any reason.
    """
    fallback = ["stream_query", "query"]
    try:
        from vertexai import agent_engines  # type: ignore

        agent = agent_engines.get(resource_id.strip())
        schemas_fn = getattr(agent, "operation_schemas", None)
        if not callable(schemas_fn):
            return fallback
        try:
            schemas = schemas_fn()
        except Exception:
            return fallback
        names: List[str] = []
        for sch in schemas or []:
            if isinstance(sch, dict):
                name = sch.get("name") or sch.get("operation_id") or sch.get("method_name")
                if isinstance(name, str) and name:
                    names.append(name)
            else:
                name = getattr(sch, "name", None) or getattr(sch, "method_name", None)
                if isinstance(name, str) and name:
                    names.append(name)
        return names or fallback
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Agent response parsing
#
# The Vertex Agent Runtime / ADK streams responses in many envelope
# shapes depending on the deployed agent and SDK version. The three
# helpers below are the canonical parsing pipeline. Transport code calls
# `_extract_agent_json(payload)` and gets back a flat dict ready for the
# schema validator, or None.
# ---------------------------------------------------------------------------

# Keys that, if present in a dict, mean we found the agent's structured
# output and should stop the recursive search.
_AGENT_SCHEMA_KEYS = {
    "occasion",
    "sub_intent",
    "formality",
    "style_direction",
    "required_slots",
    "avoid_items",
    "palette_direction",
    "category",          # metadata validator schema
    "style_role",        # metadata validator schema
}


def _strip_json_fence(text: str) -> str:
    """Drop ```json ... ``` (or plain ``` ... ```) wrappers."""
    if not isinstance(text, str):
        return text
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json|JSON)?\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _try_parse_json_text(text: Any) -> Optional[Any]:
    """Best-effort parse. Returns dict/list/None.

    - Strips ```json fences
    - Falls back to lifting the first {...} block when surrounding prose
      breaks json.loads
    """
    if not isinstance(text, str):
        return None
    raw = _strip_json_fence(text)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _looks_like_agent_dict(d: Any) -> bool:
    """True when a dict has at least one AHVI schema key."""
    if not isinstance(d, dict):
        return False
    return any(k in _AGENT_SCHEMA_KEYS for k in d.keys())


def _extract_agent_json(payload: Any, _depth: int = 0) -> Optional[Dict[str, Any]]:
    """Recursively walk any agent response envelope and return the first
    dict that matches the AHVI schema.

    Supports envelopes:
      A) Direct dict already matching the schema.
      B) ADK Runtime: {"content": {"parts": [{"text": "```json {...}```"}]}}
      C) {"output": "```json {...} ```"}
      D) {"response": "```json {...} ```"} / {"response": {"text": ...}}
      E) {"candidates": [{"content": {"parts": [...]}}]} (Gemini envelope)
      F) Arbitrarily nested dict/list combinations of the above.
    """
    if _depth > 8 or payload is None:
        return None

    # Direct match — early exit.
    if isinstance(payload, dict) and _looks_like_agent_dict(payload):
        return {k: v for k, v in payload.items() if not k.startswith("_")}

    if isinstance(payload, str):
        parsed = _try_parse_json_text(payload)
        if parsed is None:
            return None
        return _extract_agent_json(parsed, _depth + 1)

    if isinstance(payload, list):
        # Walk newest-first so the assistant's final answer wins over
        # intermediate reasoning chunks when the agent streams events.
        for item in reversed(payload):
            found = _extract_agent_json(item, _depth + 1)
            if found is not None:
                return found
        return None

    if isinstance(payload, dict):
        # Common single-text fields the agent embeds JSON into.
        for key in ("text", "output", "response", "content", "message", "result"):
            if key not in payload:
                continue
            inner = payload[key]
            # If it's a plain string with embedded JSON, parse + recurse.
            if isinstance(inner, str):
                found = _extract_agent_json(inner, _depth + 1)
                if found is not None:
                    return found
            elif isinstance(inner, (dict, list)):
                found = _extract_agent_json(inner, _depth + 1)
                if found is not None:
                    return found

        # ADK message envelope: content.parts[*].text
        content = payload.get("content")
        if isinstance(content, dict):
            parts = content.get("parts")
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, dict):
                        text = p.get("text")
                        if isinstance(text, str):
                            found = _extract_agent_json(text, _depth + 1)
                            if found is not None:
                                return found

        # Gemini candidates envelope.
        for key in ("candidates", "predictions", "outputs", "events", "_chunks"):
            inner = payload.get(key)
            if isinstance(inner, list):
                found = _extract_agent_json(inner, _depth + 1)
                if found is not None:
                    return found

        # Last-ditch: walk all remaining dict/list/string values.
        for v in payload.values():
            if isinstance(v, (dict, list, str)):
                found = _extract_agent_json(v, _depth + 1)
                if found is not None:
                    return found

    return None


def _proto_to_python(msg: Any) -> Optional[Any]:
    """Convert a gapic / protobuf chunk to a plain Python value.

    Handles three shapes seen on stream_query_reasoning_engine:
    1. `proto-plus` messages (have a `.to_dict()` classmethod or instance).
    2. Raw protobuf messages (use google.protobuf.json_format.MessageToDict).
    3. Already-decoded dicts / strings / lists.
    """
    if msg is None:
        return None
    if isinstance(msg, (dict, list, str, int, float, bool)):
        return msg
    # proto-plus instance method.
    to_dict_method = getattr(type(msg), "to_dict", None)
    if callable(to_dict_method):
        try:
            return type(msg).to_dict(msg)
        except Exception:
            pass
    # Plain google.protobuf.
    try:
        from google.protobuf.json_format import MessageToDict  # type: ignore

        return MessageToDict(msg, preserving_proto_field_name=True)
    except Exception:
        pass
    # Last resort.
    try:
        return dict(msg)  # type: ignore[arg-type]
    except Exception:
        return None


def _extract_assistant_text(chunks: List[Any]) -> str:
    """Walk ADK stream_query chunks and return the last assistant text.

    Each chunk after MessageToDict looks like:
      {"output": {"content": "..."}}                          # generic
      {"content": {"parts": [{"text": "..."}], "role": "..."}} # ADK
      {"output": {"parts": [{"text": "..."}]}}                # variant
    """
    last_text = ""
    for ch in chunks or []:
        if not isinstance(ch, dict):
            continue
        # Top-level content.
        content = ch.get("content")
        if isinstance(content, dict):
            parts = content.get("parts")
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"].strip():
                        last_text = p["text"]
        if isinstance(content, str) and content.strip():
            last_text = content
        # output.content / output.parts.
        output = ch.get("output")
        if isinstance(output, dict):
            inner = output.get("content")
            if isinstance(inner, str) and inner.strip():
                last_text = inner
            if isinstance(inner, dict):
                parts = inner.get("parts")
                if isinstance(parts, list):
                    for p in parts:
                        if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"].strip():
                            last_text = p["text"]
            outer_parts = output.get("parts")
            if isinstance(outer_parts, list):
                for p in outer_parts:
                    if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"].strip():
                        last_text = p["text"]
        # Plain text field.
        if isinstance(ch.get("text"), str) and ch["text"].strip():
            last_text = ch["text"]
    return last_text


def _flatten(value: Any, limit: int = 500) -> str:
    """Single-line, length-bounded representation for log fields.

    Cloud Logging splits stdout on `\\n`, so any newline in str(exc)
    cuts off the rest of the warning line and we lose detail/class_method/
    input_keys.
    """
    text = "" if value is None else str(value)
    return text.replace("\n", " | ").replace("\r", " ")[:limit]


def _log_gapic_detail(exc: Exception, req: Dict[str, Any]) -> None:
    """Surface the underlying gRPC/REST error detail.

    Vertex wraps the real reason (e.g. "Operation 'stream_query' not
    registered" or "Schema validation failed: missing field 'X'") inside
    a generic '400 Reasoning Engine Execution failed.' string. Pull the
    detail off the exception so we can iterate without re-deploying.
    """
    detail = ""
    for attr in ("details", "message", "reason"):
        val = getattr(exc, attr, None)
        try:
            if callable(val):
                val = val()
            if val:
                detail = str(val)
                break
        except Exception:
            continue
    # repr(exc) keeps everything on one line including the chained cause
    # which often carries the real schema error.
    logger.warning(
        "ahvi.agent.reasoning_engine_gapic_attempt_failed "
        "err=%s err_type=%s detail=%s repr=%s class_method=%s input_keys=%s",
        _flatten(exc, 300),
        type(exc).__name__,
        _flatten(detail, 600),
        _flatten(repr(exc), 600),
        (req.get("input") or {}).get("class_method"),
        sorted(list(((req.get("input") or {}).get("input") or {}).keys())),
    )


def _google_bearer_token() -> Optional[str]:
    """Mint a bearer token via Application Default Credentials."""
    try:
        from google.auth import default as google_auth_default  # type: ignore
        from google.auth.transport.requests import Request as GAuthRequest  # type: ignore

        creds, _ = google_auth_default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        if not creds.valid:
            creds.refresh(GAuthRequest())
        return getattr(creds, "token", None)
    except Exception as exc:
        logger.warning(
            "ahvi.agent.reasoning_engine_auth_failed err=%s", _flatten(exc, 200)
        )
        return None


def _adk_invoke_via_rest(
    resource_id: str,
    *,
    location: str,
    prompt: str,
    user_id: str = "ahvi_backend",
    timeout: float = 20.0,
) -> Optional[Dict[str, Any]]:
    """Invoke a deployed ADK Agent Runtime via the v1 REST API.

    Per https://adk.dev/deploy/agent-runtime/test/ the dispatch is:
      Step 1: POST {resource}:query
              { "class_method": "async_create_session",
                "input": {"user_id": ...} }
              -> {"output": {"id": "<session_id>", ...}}
      Step 2: POST {resource}:streamQuery?alt=sse
              { "class_method": "async_stream_query",
                "input": {"user_id": ..., "session_id": ..., "message": ...} }
              -> Server-Sent Events stream of agent events.
    """
    try:
        import requests  # type: ignore  (google-auth pulls it in)
    except Exception as exc:
        logger.warning(
            "ahvi.agent.reasoning_engine_requests_unavailable err=%s",
            _flatten(exc, 200),
        )
        return None

    token = _google_bearer_token()
    if not token:
        return None

    base = f"https://{location}-aiplatform.googleapis.com/v1/{resource_id.strip()}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # --- Step 1: create session ---
    # Runtime operation names vary: try `create_session` first (current
    # deployed agent ships this) and fall back to `async_create_session`.
    session_id: Optional[str] = None
    session_method_used: Optional[str] = None
    for sess_method in ("create_session", "async_create_session"):
        try:
            sess_resp = requests.post(
                f"{base}:query",
                json={
                    "class_method": sess_method,
                    "input": {"user_id": user_id},
                },
                headers=headers,
                timeout=max(5.0, timeout),
            )
        except Exception as exc:
            logger.warning(
                "ahvi.agent.reasoning_engine_session_request_failed "
                "method=%s err=%s",
                sess_method, _flatten(exc, 200),
            )
            continue

        if sess_resp.status_code >= 400:
            logger.warning(
                "ahvi.agent.reasoning_engine_session_http_error "
                "method=%s status=%d body=%s",
                sess_method, sess_resp.status_code,
                _flatten(sess_resp.text, 400),
            )
            continue

        try:
            sess_data = sess_resp.json()
        except Exception:
            logger.warning(
                "ahvi.agent.reasoning_engine_session_bad_json method=%s body=%s",
                sess_method, _flatten(sess_resp.text, 400),
            )
            continue

        output = sess_data.get("output") if isinstance(sess_data, dict) else None
        if isinstance(output, dict):
            session_id = output.get("id") or output.get("session_id")
        elif isinstance(sess_data, dict):
            session_id = sess_data.get("id") or sess_data.get("session_id")
        if session_id:
            session_method_used = sess_method
            break
        logger.warning(
            "ahvi.agent.reasoning_engine_session_missing_id method=%s body=%s",
            sess_method, _flatten(sess_data, 400),
        )

    if session_id:
        logger.info(
            "ahvi.agent.reasoning_engine_session_created "
            "resource=%s method=%s session_id=%s",
            resource_id, session_method_used, session_id,
        )

    # --- Step 2: stream the query ---
    # Build attempt list. When session was created, send with full envelope
    # (ADK-style message object first, then plain string fallback). If
    # session creation failed, still try stream_query without a session id.
    adk_message_obj = {"role": "user", "parts": [{"text": prompt}]}
    base_payloads: List[Dict[str, Any]] = []
    if session_id:
        base_payloads.extend([
            {
                "class_method": "stream_query",
                "input": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "message": adk_message_obj,
                },
            },
            {
                "class_method": "stream_query",
                "input": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "message": prompt,
                },
            },
        ])
    base_payloads.extend([
        {
            "class_method": "stream_query",
            "input": {"message": adk_message_obj},
        },
        {
            "class_method": "stream_query",
            "input": {"message": prompt},
        },
    ])

    stream_resp = None
    for body in base_payloads:
        logger.info(
            "ahvi.agent.reasoning_engine_stream_attempt class_method=%s input_keys=%s",
            body.get("class_method"),
            sorted(list((body.get("input") or {}).keys())),
        )
        try:
            stream_resp = requests.post(
                f"{base}:streamQuery?alt=sse",
                json=body,
                headers=headers,
                timeout=max(10.0, timeout),
                stream=True,
            )
        except Exception as exc:
            logger.warning(
                "ahvi.agent.reasoning_engine_stream_request_failed "
                "class_method=%s err=%s",
                body.get("class_method"), _flatten(exc, 200),
            )
            stream_resp = None
            continue

        if stream_resp.status_code >= 400:
            logger.warning(
                "ahvi.agent.reasoning_engine_stream_http_error "
                "class_method=%s input_keys=%s status=%d body=%s",
                body.get("class_method"),
                sorted(list((body.get("input") or {}).keys())),
                stream_resp.status_code,
                _flatten(stream_resp.text, 400),
            )
            stream_resp = None
            continue
        break

    if stream_resp is None:
        return None

    logger.info(
        "ahvi.agent.reasoning_engine_stream_call resource=%s session_id=%s",
        resource_id,
        session_id or "<none>",
    )

    # --- Step 3: drain the stream + collect event JSON ---
    # The ADK runtime sometimes responds in SSE shape (`data: {...}`) and
    # sometimes streams raw JSON lines (no `data:` prefix) even when
    # ?alt=sse is requested. Handle both. Also try buffering the full
    # body and parsing as one JSON blob when line-by-line parsing yields
    # nothing.
    events: List[Any] = []
    raw_lines: List[str] = []
    try:
        for raw_line in stream_resp.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            raw_lines.append(raw_line)
            if not raw_line:
                continue
            payload = raw_line
            if payload.startswith("data:"):
                payload = payload[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                # Not parseable on its own — keep as text fragment, the
                # full-body fallback below will try concatenated JSON.
                events.append(payload)
    except Exception as exc:
        logger.warning(
            "ahvi.agent.reasoning_engine_stream_iter_failed err=%s",
            _flatten(exc, 200),
        )
        return None

    # Fallback: some streams emit a single JSON object split across many
    # iter_lines chunks. If we got no parsed events but have raw lines,
    # try parsing the concatenated body.
    parsed_dict_events = [e for e in events if isinstance(e, dict)]
    if not parsed_dict_events and raw_lines:
        joined = "".join(
            line[5:].strip() if line.startswith("data:") else line
            for line in raw_lines
        ).strip()
        if joined and joined != "[DONE]":
            try:
                events.append(json.loads(joined))
            except json.JSONDecodeError:
                # Try to lift the first {...} block (helpful when the
                # body has leading log noise or trailing commentary).
                m = re.search(r"\{.*\}", joined, re.DOTALL)
                if m:
                    try:
                        events.append(json.loads(m.group(0)))
                    except Exception:
                        pass

    logger.info(
        "ahvi.agent.reasoning_engine_stream_raw lines=%d events=%d preview=%s",
        len(raw_lines),
        len(events),
        _flatten(raw_lines[-3:] if raw_lines else [], 300),
    )

    if not events:
        return None

    # Recursive extractor handles every envelope we've seen so far:
    # ADK content.parts[*].text with ```json fences, direct dict,
    # {output:...}, {response:...}, {candidates:[...]} etc.
    parsed = _extract_agent_json(events)
    if isinstance(parsed, dict) and parsed:
        logger.info(
            "ahvi.agent.reasoning_engine_parse_success keys=%s",
            sorted(parsed.keys())[:12],
        )
        return parsed

    preview = _flatten(events[-1] if events else raw_lines, 500)
    logger.warning(
        "ahvi.agent.reasoning_engine_parse_failed preview=%s",
        preview,
    )
    return None


def _invoke_via_gapic(
    resource_id: str,
    *,
    location: str,
    prompt: str,
    user_id: str = "ahvi-backend",
    preferred_class_methods: Optional[List[str]] = None,
) -> Optional[Any]:
    """Call the deployed Agent Engine through the gapic execution client.

    The high-level vertexai handle (`AgentEngine`) only exposes management
    operations (create/delete/update/operation_schemas). The actual
    `query` / `stream_query` operations on a deployed ADK agent live on
    `ReasoningEngineExecutionServiceClient` in google-cloud-aiplatform.
    """
    try:
        from google.cloud import aiplatform_v1beta1 as aiplatform_gapic  # type: ignore
    except Exception as exc:
        logger.warning(
            "ahvi.agent.reasoning_engine_gapic_unavailable err=%s", str(exc)[:200]
        )
        return None

    api_endpoint = f"{location}-aiplatform.googleapis.com"
    try:
        client = aiplatform_gapic.ReasoningEngineExecutionServiceClient(
            client_options={"api_endpoint": api_endpoint}
        )
    except Exception as exc:
        logger.warning(
            "ahvi.agent.reasoning_engine_client_init_failed err=%s", str(exc)[:200]
        )
        return None

    # Vertex Agent Engine dispatches to the deployed agent's registered
    # operations via `class_method`. For ADK LlmAgents the live operations
    # are `stream_query` (event stream) and sometimes `query` (single
    # response). Each gets the agent's input under nested `input.input`.
    adk_inputs = [
        {"message": prompt, "user_id": user_id},
        {"input": prompt, "user_id": user_id},
        {"prompt": prompt, "user_id": user_id},
        {"message": prompt},
    ]

    def _build_request(class_method: str, agent_input: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": resource_id,
            "input": {
                "class_method": class_method,
                "input": agent_input,
            },
        }

    # Prefer streaming because that's how ADK Agent Engines expose
    # LlmAgent.run output; fall back to non-streaming `query` if present.
    last_exc: Optional[Exception] = None
    discovered = list(preferred_class_methods or [])
    # Build a (rpc, class_method) probe matrix. Use discovered operations
    # first so we don't waste calls on names the agent never registered.
    if discovered:
        stream_methods = [m for m in discovered if "stream" in m.lower()]
        nonstream_methods = [m for m in discovered if "stream" not in m.lower()]
    else:
        stream_methods = ["stream_query"]
        nonstream_methods = ["query"]

    probes = []
    for cm in stream_methods:
        probes.append(("stream_query_reasoning_engine", cm))
    for cm in nonstream_methods:
        probes.append(("query_reasoning_engine", cm))

    for call_name, method_short in probes:
        method = getattr(client, call_name, None)
        if not callable(method):
            continue
        request_payloads = [_build_request(method_short, agent_input) for agent_input in adk_inputs]
        for req in request_payloads:
            try:
                resp = method(request=req)
            except TypeError:
                try:
                    resp = method(**req)
                except Exception as exc:
                    last_exc = exc
                    _log_gapic_detail(exc, req)
                    continue
            except Exception as exc:
                last_exc = exc
                _log_gapic_detail(exc, req)
                continue

            # Streaming → iterate. Non-streaming → single response.
            if hasattr(resp, "__iter__") and not isinstance(resp, (dict, str, bytes)):
                chunks: List[Any] = []
                try:
                    for ch in resp:
                        chunk_py = _proto_to_python(ch)
                        if chunk_py is not None:
                            chunks.append(chunk_py)
                except Exception as exc:
                    last_exc = exc
                    continue
                if chunks:
                    logger.info(
                        "ahvi.agent.reasoning_engine_gapic_chunks resource=%s count=%d sample_keys=%s",
                        resource_id,
                        len(chunks),
                        sorted(list(chunks[-1].keys()))[:8] if isinstance(chunks[-1], dict) else [],
                    )
                    return {
                        "_method": call_name,
                        "_chunks": chunks,
                        "output": {"content": _extract_assistant_text(chunks)},
                    }
            else:
                py = _proto_to_python(resp)
                if py is not None:
                    if isinstance(py, dict):
                        py.setdefault("_method", call_name)
                    return py
                return {"output": {"content": str(resp)}, "_method": call_name}

    if last_exc is not None:
        logger.warning(
            "ahvi.agent.reasoning_engine_gapic_call_failed resource=%s err=%s "
            "last_request_keys=%s",
            resource_id,
            str(last_exc)[:300],
            sorted(list((request_payloads[-1].get("input") or {}).keys())) if request_payloads else [],
        )
    return None


def _invoke_reasoning_engine_sync(
    resource_id: str,
    *,
    system: str,
    prompt: str,
) -> Optional[Dict[str, Any]]:
    """Synchronous Reasoning Engine call.

    Wrapped in `asyncio.to_thread` by the public async helper so the
    surrounding event loop is not blocked.
    """
    try:
        import vertexai  # type: ignore  # noqa: F401  (kept for SDK init)
    except Exception as exc:
        logger.warning("ahvi.agent.reasoning_engine_sdk_unavailable err=%s", str(exc)[:200])
        return None

    project = _project_from_env()
    location = _location_from_resource(resource_id, fallback="us-central1")

    try:
        try:
            import vertexai  # type: ignore
            if project:
                vertexai.init(project=project, location=location)
        except Exception:
            pass

        # PRIMARY PATH: ADK Agent Runtime REST flow (per adk.dev docs).
        # Two POSTs: async_create_session then async_stream_query against
        # the deployed reasoning engine. This is the only dispatch shape
        # the runtime accepts for ADK-deployed LlmAgents.
        adk_result = _adk_invoke_via_rest(
            resource_id,
            location=location,
            prompt=prompt,
        )
        if isinstance(adk_result, dict) and adk_result:
            logger.info(
                "ahvi.agent.reasoning_engine_call_ok resource=%s method=adk_rest_stream_query",
                resource_id,
            )
            return adk_result

        # FALLBACK: discover operation names + gapic execution client.
        # Used when the REST flow returns nothing (custom non-ADK engine).
        registered_ops = _discover_operation_names(resource_id)
        logger.info(
            "ahvi.agent.reasoning_engine_operations resource=%s ops=%s",
            resource_id,
            registered_ops,
        )

        gapic_result = _invoke_via_gapic(
            resource_id,
            location=location,
            prompt=prompt,
            preferred_class_methods=registered_ops,
        )
        if gapic_result is not None:
            logger.info(
                "ahvi.agent.reasoning_engine_call_ok resource=%s method=%s",
                resource_id,
                gapic_result.get("_method") if isinstance(gapic_result, dict) else "gapic",
            )
            return _parse_engine_payload(gapic_result)

        # FALLBACK PATH: try the high-level handle's bound methods in
        # case the SDK is newer and exposes them directly.
        engine = _load_engine(resource_id)
        if engine is None:
            logger.warning(
                "ahvi.agent.reasoning_engine_load_failed resource=%s reason=no_sdk_handle",
                resource_id,
            )
            return None

        all_callables = _list_engine_callables(engine)
        logger.info(
            "ahvi.agent.reasoning_engine_loaded resource=%s type=%s callables=%s",
            resource_id,
            type(engine).__name__,
            all_callables[:30],
        )

        try:
            raw, method = _call_engine_with_fallbacks(engine, system, prompt)
        except Exception as exc:
            logger.warning(
                "ahvi.agent.reasoning_engine_call_failed resource=%s err=%s "
                "probed=%s available_methods=%s engine_type=%s",
                resource_id,
                str(exc)[:200],
                list(_METHOD_NAMES),
                all_callables,
                type(engine).__name__,
            )
            return None

        logger.info(
            "ahvi.agent.reasoning_engine_call_ok resource=%s method=%s",
            resource_id,
            method,
        )
        return _parse_engine_payload(raw)
    except Exception as exc:
        logger.warning(
            "ahvi.agent.reasoning_engine_call_failed resource=%s err=%s",
            resource_id,
            str(exc)[:200],
        )
        return None


async def call_reasoning_engine(
    resource_id: str,
    *,
    system: str,
    prompt: str,
    timeout: float = 12.0,
) -> Optional[Dict[str, Any]]:
    """Async wrapper. Returns parsed dict or None on any failure."""
    if not looks_like_resource_id(resource_id):
        return None
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _invoke_reasoning_engine_sync,
                resource_id,
                system=system,
                prompt=prompt,
            ),
            timeout=max(1.0, float(timeout)),
        )
    except asyncio.TimeoutError:
        logger.warning("ahvi.agent.reasoning_engine_timeout resource=%s", resource_id)
        return None
    except Exception as exc:
        logger.warning(
            "ahvi.agent.reasoning_engine_unexpected resource=%s err=%s",
            resource_id,
            str(exc)[:200],
        )
        return None


__all__ = ["call_reasoning_engine", "looks_like_resource_id"]
