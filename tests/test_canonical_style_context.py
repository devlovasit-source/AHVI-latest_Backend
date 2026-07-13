from types import SimpleNamespace

from brain.engines.style_scorer import _memory_breakdown, score_weather_compatibility
from routers import chat, stylist
from services import style_context_service as scs
from services import style_memory_service as sms
from services.style_flow_service import _style_dna_alignment_text


WARDROBE = [
    {"id": "top-1", "name": "Linen shirt", "category": "top"},
    {"id": "bottom-1", "name": "Chinos", "category": "bottom"},
    {"id": "shoe-1", "name": "Sneakers", "category": "footwear"},
]
PROFILE = {
    "gender": "male",
    "style_dna": {
        "style_archetypes": {"minimal": 0.9},
        "color_dna": {"core_colors": ["navy"]},
    },
    "weather": {"condition": "hot", "temp_c": 31},
}
MEMORY = {
    "recently_worn_ids": ["top-1"],
    "underworn_ids": ["bottom-1"],
    "wear_counts": {"top-1": 3, "bottom-1": 1},
    "saved_item_ids": ["shoe-1"],
    "favorite_colors": ["navy"],
}


def _http_request(user_id="user-1"):
    return SimpleNamespace(
        state=SimpleNamespace(user={"user_id": user_id}),
        url=SimpleNamespace(path="/test"),
    )


def _empty_flow(**kwargs):
    return {"success": True, "cards": [], "meta": {}}


def test_chat_and_stylist_pipeline_share_equivalent_intelligence_context(monkeypatch):
    captured = {}
    monkeypatch.setattr(chat, "_ahvi_resolve_effective_user_profile", lambda *_: PROFILE)
    monkeypatch.setattr(chat, "_fetch_wardrobe_for_style", lambda *_: WARDROBE)
    monkeypatch.setattr(chat, "_ahvi_item_allowed_for_user_profile", lambda *_: True)
    monkeypatch.setattr(sms, "build_style_memory_context", lambda *_: MEMORY)
    monkeypatch.setattr(
        chat,
        "build_style_flow_response",
        lambda **kwargs: captured.setdefault("chat", kwargs["context"]) or _empty_flow(),
    )
    monkeypatch.setattr(
        stylist,
        "build_style_flow_response",
        lambda **kwargs: captured.setdefault("stylist", kwargs["context"]) or _empty_flow(),
    )
    monkeypatch.setattr("services.data_access_service.get_user_profile", lambda **_: PROFILE)

    chat._demo_style_board_payload("user-1", "office meeting", WARDROBE, PROFILE)
    stylist.run_outfit_pipeline(
        stylist.OutfitPipelineRequest(
            query="office meeting",
            wardrobe=WARDROBE,
            user_profile=PROFILE,
            context={"weather": PROFILE["weather"]},
        ),
        _http_request(),
    )

    intelligence_keys = (
        "canonical_occasion", "occasion_brief", "gender", "style_dna",
        "weather_context", "event_context", "wardrobe_summary",
        "recently_worn_ids", "underworn_ids", "wear_counts", "saved_item_ids",
    )
    assert {key: captured["chat"][key] for key in intelligence_keys} == {
        key: captured["stylist"][key] for key in intelligence_keys
    }


def test_chat_propagates_profile_style_dna_to_style_flow(monkeypatch):
    captured = {}
    monkeypatch.setattr(chat, "_ahvi_resolve_effective_user_profile", lambda *_: PROFILE)
    monkeypatch.setattr(chat, "_fetch_wardrobe_for_style", lambda *_: WARDROBE)
    monkeypatch.setattr(chat, "_ahvi_item_allowed_for_user_profile", lambda *_: True)
    monkeypatch.setattr(sms, "build_style_memory_context", lambda *_: {})
    monkeypatch.setattr(
        chat,
        "build_style_flow_response",
        lambda **kwargs: captured.setdefault("context", kwargs["context"]) or _empty_flow(),
    )

    chat._demo_style_board_payload("user-1", "office meeting", WARDROBE, PROFILE)

    assert captured["context"]["style_dna"]["style_archetypes"] == ["minimal"]
    assert captured["context"]["context_provenance"]["style_dna_used"] is True


def test_structured_hot_and_rain_weather_reaches_scoring():
    outfit = {"items": [{"name": "Linen shirt", "category": "top"}]}

    hot = score_weather_compatibility(
        outfit, {"weather_context": {"condition": "clear", "temperature_c": 31}}
    )
    rain = score_weather_compatibility(
        outfit, {"weather_context": {"condition": "rain", "precipitation": 0.8}}
    )

    assert hot["weather"] == "summer"
    assert rain["weather"] == "rain"


def test_missing_weather_is_neutral():
    result = score_weather_compatibility({"items": []}, {"weather_context": {}})

    assert result["score"] == 0.5
    assert result["raw_score"] == 0.0
    assert result["reason"] == "weather_unknown"


def test_durable_memory_signals_keep_existing_penalty_and_boosts():
    fields, _ = _memory_breakdown(
        WARDROBE,
        {
            "recently_worn_ids": ["top-1"],
            "underworn_ids": ["bottom-1"],
            "saved_item_ids": ["shoe-1"],
        },
    )

    assert fields["recent_repeat_penalty"] == -1.5
    assert fields["underworn_boost"] == 1.2
    assert fields["saved_board_affinity"] == 1.0


def test_empty_memory_has_no_automatic_positive_score(monkeypatch):
    class EmptyProxy:
        def list_documents(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(sms, "_proxy", lambda: EmptyProxy())
    memory = sms.load_wear_memory("user-1", WARDROBE)
    fields, _ = _memory_breakdown(WARDROBE, memory)

    assert memory["underworn_ids"] == []
    assert all(value == 0 for value in fields.values())


def test_unknown_gender_and_multi_event_chronology_remain_explicit():
    ctx = scs.build_canonical_style_context(
        query="gym at 6pm then brunch at 10pm",
        user_profile={},
        memory={},
        profile_is_authenticated=True,
    )

    assert ctx["gender"] == "unknown"
    assert ctx["canonical_occasion"] == ctx["dominant_occasion"] == "brunch"
    assert [step["event"] for step in ctx["time_sequence"]] == ["workout", "brunch"]
    assert [step["time"] for step in ctx["time_sequence"]] == ["18:00", "22:00"]
    assert ctx["context_provenance"]["canonical_occasion"] == "brunch"


def test_empty_style_dna_does_not_claim_personal_alignment():
    assert _style_dna_alignment_text(
        {"style_dna_available": False, "style_preferences": []},
        "Modern Professional",
    ) == ""
