from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from brain import outfit_pipeline as pipeline
from routers import stylist
from services.style_execution_policy import (
    ModelCallBudget,
    ModelCallBudgetExceeded,
    StyleExecutionPolicy,
    StyleExecutionSession,
    StyleExecutionSinks,
    UnknownStyleExecutionPolicy,
    activate_style_execution,
    consume_model_call,
    create_style_execution_session,
    get_style_execution_session,
    image_generation_allowed,
    run_board_registration,
    run_learning_vector_upsert,
    run_preference_memory_write,
)
from services import agent_style_orchestrator as style_agent


def _session(limit: int = 2, *, learning: bool = False, sinks=None):
    policy = StyleExecutionPolicy(
        name="read_only_evaluation",
        allow_preference_learning=learning,
        allow_board_registration=False,
        allow_cache_writes=False,
        allow_image_generation=False,
        model_call_limit=limit,
    )
    return StyleExecutionSession(policy, ModelCallBudget(limit), sinks or StyleExecutionSinks())


def test_unknown_policy_fails_closed():
    with pytest.raises(UnknownStyleExecutionPolicy):
        create_style_execution_session("caller_selected")


def test_read_only_evaluation_has_strict_budget_and_disables_images():
    session = create_style_execution_session("read_only_evaluation")
    with activate_style_execution(session):
        assert session.budget.limit == 1
        assert image_generation_allowed() is False
        assert session.policy.allow_board_registration is False


def test_read_only_evaluation_does_not_mutate_computation_caches(monkeypatch):
    pipeline._QUERY_VECTOR_CACHE.clear()
    monkeypatch.setattr(
        pipeline,
        "embedding_service",
        SimpleNamespace(enabled=True, encode_text=lambda *_: [0.1, 0.2]),
    )
    style_agent._AGENT_CACHE.clear()
    style_agent._AGENT_CACHE["expired"] = (0.0, {"confidence": 0.9})

    with activate_style_execution(create_style_execution_session("read_only_evaluation")):
        assert pipeline._cached_query_vector("hot weather") == [0.1, 0.2]
        assert style_agent._agent_cache_get("expired") is None

    assert pipeline._QUERY_VECTOR_CACHE == {}
    assert "expired" in style_agent._AGENT_CACHE
    style_agent._AGENT_CACHE.clear()


def test_nested_and_parallel_stages_share_one_budget():
    session = _session(2)

    def worker():
        with activate_style_execution(session):
            return consume_model_call(stage="combination_filter", model_alias="safe-model")

    with activate_style_execution(session):
        assert consume_model_call(stage="orchestration", model_alias="safe-model") == 1
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(worker).result() == 2
        with pytest.raises(ModelCallBudgetExceeded):
            consume_model_call(stage="curation", model_alias="safe-model")
    assert session.budget.count == 2


def test_retry_consumes_budget_and_exhaustion_prevents_call():
    session = _session(2)
    calls = []

    def attempt(stage):
        consume_model_call(stage=stage, model_alias="safe-model")
        calls.append(stage)

    with activate_style_execution(session):
        attempt("curation")
        attempt("curation.retry")
        with pytest.raises(ModelCallBudgetExceeded):
            attempt("fallback")
    assert calls == ["curation", "curation.retry"]


def test_exhausted_optional_combo_model_uses_deterministic_fallback(monkeypatch):
    session = _session(0)
    calls = []

    def blocked_model(*args, **kwargs):
        consume_model_call(stage="combination_filter", model_alias="safe-model")
        calls.append("called")
        return {"selected_combo_ids": ["combo-1"]}

    monkeypatch.setattr(pipeline.ai_gateway, "generate_json_object", blocked_model)
    combo = {
        "combo_id": "combo-1",
        "top": {"id": "top-1", "name": "White Shirt"},
        "bottom": {"id": "bottom-1", "name": "Navy Chinos"},
        "shoes": {"id": "shoe-1", "name": "Brown Loafers"},
    }
    with activate_style_execution(session):
        selected = pipeline._llm_filter_combo_ids(
            occasion="daily",
            stage="color_combo",
            master_type="top",
            master_piece=combo["top"],
            combos=[combo],
        )
    assert selected == []
    assert calls == []


def test_generation_policy_blocks_preference_and_vector_sinks():
    calls = []
    sinks = StyleExecutionSinks(
        preference_memory=lambda *a, **k: calls.append("memory"),
        learning_vector=lambda *a, **k: calls.append("vector"),
    )
    with activate_style_execution(_session(sinks=sinks)):
        assert run_preference_memory_write(lambda: calls.append("default-memory")) is False
        assert run_learning_vector_upsert(lambda: calls.append("default-vector")) is None
    assert calls == []


def test_explicit_action_outside_generation_persists_each_sink_once():
    calls = []
    assert run_preference_memory_write(lambda: calls.append("memory") or True) is True
    run_learning_vector_upsert(lambda: calls.append("vector"))
    assert calls == ["memory", "vector"]


def test_immutable_board_registration_remains_injectable():
    calls = []
    sinks = StyleExecutionSinks(board_state=lambda **kwargs: calls.append(kwargs) or {"ok": True})
    production = StyleExecutionPolicy(
        name="production",
        allow_preference_learning=False,
        allow_board_registration=True,
        allow_cache_writes=True,
        allow_image_generation=True,
        model_call_limit=2,
    )
    with activate_style_execution(StyleExecutionSession(production, ModelCallBudget(2), sinks)):
        result = run_board_registration(lambda **kwargs: {"ok": False}, board_id="board-1")
    assert result == {"ok": True}
    assert calls == [{"board_id": "board-1"}]


def test_caller_context_cannot_select_policy_or_raise_budget(monkeypatch):
    seen = {}
    monkeypatch.setattr(stylist, "bind_request_user", lambda *_: "user-1")
    monkeypatch.setattr(stylist, "build_canonical_style_context", lambda **kwargs: kwargs["request_context"])

    def fake_flow(**kwargs):
        active = get_style_execution_session()
        seen.update(name=active.policy.name, limit=active.budget.limit)
        return {"meta": {}}

    monkeypatch.setattr(stylist, "build_style_flow_response", fake_flow)
    request = stylist.OutfitPipelineRequest(
        user_id="user-1",
        wardrobe=[],
        context={"execution_policy": "read_only_evaluation", "model_call_budget": 999},
    )
    stylist.run_outfit_pipeline(
        request,
        SimpleNamespace(state=SimpleNamespace(user={"user_id": "user-1"})),
    )
    assert seen == {"name": "production", "limit": 3}


def test_actual_generation_does_not_write_learning_sinks(monkeypatch):
    memory_calls, vector_calls = [], []
    monkeypatch.setenv("ENABLE_AGENT_STYLE_ORCHESTRATOR", "false")
    monkeypatch.setenv("OUTFIT_LLM_COMBO_FILTER", "false")
    monkeypatch.setattr(pipeline, "_load_user_memory", lambda *_: pipeline._default_user_memory())
    monkeypatch.setattr(pipeline, "_save_user_memory", lambda *a, **k: memory_calls.append((a, k)))
    monkeypatch.setattr(pipeline, "_index_outfit_vector", lambda *a, **k: vector_calls.append((a, k)))
    wardrobe = [
        {"id": "top-1", "name": "White Shirt", "category": "Tops", "color": "white"},
        {"id": "bottom-1", "name": "Navy Chinos", "category": "Bottoms", "color": "navy"},
        {"id": "shoe-1", "name": "Brown Loafers", "category": "Footwear", "color": "brown"},
    ]
    result = pipeline.get_daily_outfits(
        {"user_id": "user-1", "wardrobe": wardrobe, "context": {"occasion": "daily"}}
    )
    assert isinstance(result, dict)
    assert memory_calls == []
    assert vector_calls == []
