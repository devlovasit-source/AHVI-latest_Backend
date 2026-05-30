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


def _invoke_via_gapic(
    resource_id: str,
    *,
    location: str,
    prompt: str,
    user_id: str = "ahvi-backend",
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

    # The ADK agent's stream_query operation takes
    # input.message + input.user_id + input.session_id (session optional).
    request_payloads = [
        {
            "name": resource_id,
            "input": {"message": prompt, "user_id": user_id},
        },
        {
            "name": resource_id,
            "input": {"input": prompt},
        },
        {
            "name": resource_id,
            "input": {"prompt": prompt},
        },
    ]

    # Prefer streaming because that's how ADK Agent Engines expose
    # LlmAgent.run output; fall back to non-streaming `query` if present.
    last_exc: Optional[Exception] = None
    for call_name in ("stream_query_reasoning_engine", "query_reasoning_engine"):
        method = getattr(client, call_name, None)
        if not callable(method):
            continue
        for req in request_payloads:
            try:
                resp = method(request=req)
            except TypeError:
                try:
                    resp = method(**req)
                except Exception as exc:
                    last_exc = exc
                    continue
            except Exception as exc:
                last_exc = exc
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
            "ahvi.agent.reasoning_engine_gapic_call_failed resource=%s err=%s",
            resource_id,
            str(last_exc)[:200],
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

        # PRIMARY PATH: gapic ReasoningEngineExecutionServiceClient.
        # The high-level AgentEngine handle returned by agent_engines.get()
        # only exposes management ops (create/delete/list/update). The
        # actual stream_query / query operations live on the gapic client.
        gapic_result = _invoke_via_gapic(
            resource_id, location=location, prompt=prompt
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
