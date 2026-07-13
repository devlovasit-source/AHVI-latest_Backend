import builtins
from copy import deepcopy

from brain.engines.style_scorer import _memory_breakdown
from brain.ml.outfit_ranker import BASELINE_WEIGHTS, OutfitRanker
from brain.personalization.style_dna_engine import StyleDNAEngine
from services import style_context_service as scs
from services import style_memory_service as sms


WARDROBE = [
    {"id": "Top-A", "name": "Navy shirt", "category": "top", "color": "navy"},
    {"id": "Shoe-B", "name": "Brown loafers", "category": "footwear", "color": "brown"},
]


def _durable_context():
    return {
        "user_profile": {
            "gender": "male",
            "style_preferences": {
                "archetypes": ["Minimal", "Tailored"],
                "colors": ["Navy"],
                "avoided_colors": ["Neon Green"],
            },
        },
        "wardrobe_items": deepcopy(WARDROBE),
        "memory": {
            "liked_item_ids": ["top-a"],
            "disliked_item_ids": ["shoe-b"],
            "liked_board_patterns": ["quiet luxury"],
            "disliked_board_patterns": ["maximalist"],
            "saved_board_patterns": ["business casual"],
            "saved_item_ids": ["top-a"],
            "wear_counts": {"top-a": 3},
            "recently_worn_ids": ["top-a"],
            "_personalization_meta": {"personalization_degraded": False},
        },
    }


def test_empty_inputs_produce_neutral_dna_without_casual_default():
    dna = StyleDNAEngine().build({})
    assert dna["confidence"] == 0
    assert dna["dna_signal_count"] == 0
    assert dna["style"] == dna["primary_aesthetic"] == ""
    assert "casual" not in str(dna).lower()


def test_explicit_preferences_survive_normalization():
    dna = StyleDNAEngine().build(_durable_context())
    assert dna["style_archetypes"][:2] == ["minimal", "tailored"]
    assert dna["preferred_colors"][0] == "navy"
    assert dna["avoided_colors"] == ["neon green"]
    assert dna["confidence"] > 0


def test_durable_likes_and_dislikes_influence_derived_dna():
    dna = StyleDNAEngine().build(_durable_context())
    assert "top" in dna["preferred_types"]
    assert "footwear" in dna["disliked_items"]
    assert "quiet luxury" in dna["preferred_styles"]
    assert "maximalist" in dna["avoid_style_keywords"]
    assert dna["durable_feedback_used"] is True


def test_identical_inputs_are_identical_across_restart_simulation():
    payload = _durable_context()
    first = StyleDNAEngine().build(deepcopy(payload))
    second = StyleDNAEngine().build(deepcopy(payload))
    assert first == second
    assert not hasattr(StyleDNAEngine(), "_dna_path")


def test_cross_user_feedback_never_enters_other_users_dna(monkeypatch):
    monkeypatch.setattr(sms, "load_wear_memory", lambda *_: {
        "recently_worn_ids": [], "underworn_ids": [], "wear_counts": {},
        "last_worn_at": {}, "_degraded": False,
    })
    monkeypatch.setattr(sms, "load_saved_board_memory", lambda *_: {
        "saved_item_ids": [], "saved_board_patterns": [], "favorite_colors": [],
        "favorite_categories": [], "_degraded": False,
    })
    monkeypatch.setattr(
        "services.style_feedback_store.load_feedback_memory",
        lambda user_id: {
            "liked_item_ids": ["top-a"] if user_id == "user-a" else [],
            "disliked_item_ids": [], "feedback_saved_item_ids": [],
            "liked_board_patterns": ["minimal"] if user_id == "user-a" else [],
        },
    )
    first = sms.build_style_memory_context("user-a", WARDROBE)
    second = sms.build_style_memory_context("user-b", WARDROBE)
    dna_a = StyleDNAEngine().build({"memory": first, "wardrobe_items": WARDROBE})
    dna_b = StyleDNAEngine().build({"memory": second, "wardrobe_items": WARDROBE})
    assert dna_a["durable_feedback_used"] is True
    assert dna_b["durable_feedback_used"] is False
    assert "minimal" not in dna_b["preferred_styles"]


def test_missing_durable_memory_degrades_neutrally(monkeypatch):
    class BrokenProxy:
        def list_documents(self, *_args, **_kwargs):
            raise RuntimeError("offline")

    monkeypatch.setattr(sms, "_proxy", lambda: BrokenProxy())
    monkeypatch.setattr(sms, "load_saved_board_memory", lambda *_: {
        "saved_item_ids": [], "saved_board_patterns": [], "favorite_colors": [],
        "favorite_categories": [], "_degraded": True,
    })
    monkeypatch.setattr(
        "services.style_feedback_store.load_feedback_memory",
        lambda *_: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    memory = sms.build_style_memory_context("user-a", WARDROBE)
    dna = StyleDNAEngine().build({"memory": memory, "wardrobe_items": WARDROBE})
    assert memory["_personalization_meta"]["personalization_degraded"] is True
    assert dna["personalization_degraded"] is True
    assert dna["confidence"] == 0
    assert dna["primary_aesthetic"] == ""


def test_personalization_paths_never_write_local_files(monkeypatch):
    from brain import outfit_pipeline as pipeline

    attempted = []
    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        name = str(file).lower()
        if any(token in name for token in (
            "style_dna_memory.json", "outfit_memory.json", "ranker_state.json"
        )):
            attempted.append((name, mode))
            raise AssertionError(f"personalization file access: {name} {mode}")
        return real_open(file, mode, *args, **kwargs)

    class BrokenProxy:
        def get_document(self, *_args, **_kwargs):
            raise RuntimeError("offline")

        def update_document(self, *_args, **_kwargs):
            raise RuntimeError("offline")

        def create_document(self, *_args, **_kwargs):
            raise RuntimeError("offline")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(pipeline, "AppwriteProxy", BrokenProxy)
    StyleDNAEngine().build(_durable_context())
    ranker = OutfitRanker()
    ranker.rank("user-a", [{"score": 1, "ml_features": {"memory": 1}}])
    ranker.learn_from_feedback("user-a", {"memory": 1}, "like")
    assert pipeline._load_user_memory("user-a") == pipeline._default_user_memory()
    assert pipeline._save_user_memory("user-a", pipeline._default_user_memory()) is False
    assert attempted == []
    assert not hasattr(ranker, "_state_path")


def test_legacy_feedback_wrapper_has_no_local_state_write(monkeypatch):
    from brain import outfit_pipeline as pipeline

    writes = []
    monkeypatch.setattr(pipeline, "_load_user_memory", lambda *_: pipeline._default_user_memory())
    monkeypatch.setattr(pipeline, "_save_user_memory", lambda *_: writes.append("durable") or True)
    monkeypatch.setattr(pipeline, "_index_outfit_vector", lambda **_: None)
    result = pipeline.save_feedback("user-a", {"id": "look-1", "ml_features": {}}, "up")
    assert result == {"ok": True, "feedback": "up"}
    assert writes == ["durable"]


def test_ranker_uses_immutable_baseline_and_does_not_learn_locally():
    ranker = OutfitRanker()
    outfit = {"id": "one", "score": 2.0, "ml_features": {"feedback": 1.0}}
    before = ranker.rank("user-a", [outfit])
    ranker.learn_from_feedback("user-a", {"feedback": 10.0}, "dislike")
    after = ranker.rank("user-a", [outfit])
    assert before == after
    assert dict(BASELINE_WEIGHTS)["feedback"] == 1.0


def test_batch8_feedback_scoring_bounds_are_unchanged():
    liked, _ = _memory_breakdown([{"id": "one"}], {"liked_item_ids": ["one"]})
    disliked, _ = _memory_breakdown(
        [{"id": "one"}],
        {"disliked_item_ids": ["one"], "underworn_ids": ["one"], "saved_item_ids": ["one"]},
    )
    assert liked["liked_item_affinity"] == 0.8
    assert disliked["disliked_item_penalty"] == -2.0
    assert disliked["underworn_boost"] == disliked["saved_board_affinity"] == 0


def test_canonical_context_loads_durable_memory_once(monkeypatch):
    calls = []
    monkeypatch.setattr(sms, "build_style_memory_context", lambda user_id, *_: calls.append(user_id) or {
        "liked_item_ids": ["top-a"], "disliked_item_ids": [],
        "liked_board_patterns": ["minimal"], "disliked_board_patterns": [],
        "saved_item_ids": [], "saved_board_patterns": [], "favorite_colors": [],
        "favorite_categories": [], "recently_worn_ids": [], "underworn_ids": [],
        "wear_counts": {}, "_personalization_meta": {"personalization_degraded": False},
    })
    context = scs.build_canonical_style_context(
        query="office", user_id="user-a", user_profile={}, wardrobe_items=WARDROBE,
        profile_is_authenticated=True,
    )
    assert calls == ["user-a"]
    assert context["style_dna"]["style_archetypes"] == ["minimal"]
    assert context["context_provenance"]["durable_feedback_used"] is True
