import pytest

from services.style_evaluation_guards import (
    EvaluationWriteDenied,
    EvaluationWriteGuards,
    MutationDeniedSink,
    ReadOnlyAppwrite,
    ReadOnlyCache,
    ReadOnlyR2,
    SearchOnlyQdrant,
)


class FakeClient:
    def __init__(self):
        self.calls = []

    def get_document(self, value):
        self.calls.append(("get_document", value))
        return {"value": value}

    def search(self, value):
        self.calls.append(("search", value))
        return [value]

    def get(self, value):
        self.calls.append(("get", value))
        return value

    def create_document(self, *_args, **_kwargs):
        raise AssertionError("network mutation must not be called")

    def upsert(self, *_args, **_kwargs):
        raise AssertionError("network mutation must not be called")

    def put(self, *_args, **_kwargs):
        raise AssertionError("network mutation must not be called")

    def set(self, *_args, **_kwargs):
        raise AssertionError("network mutation must not be called")


def test_appwrite_allows_document_reads_and_blocks_writes_before_delegate():
    client = FakeClient()
    guard = ReadOnlyAppwrite(client)
    assert guard.get_document("safe") == {"value": "safe"}
    with pytest.raises(EvaluationWriteDenied) as exc:
        guard.create_document("do-not-send")
    assert str(exc.value) == "STYLE_EVALUATION_WRITE_DENIED"
    assert guard.blocked_write_attempts == 1
    assert client.calls == [("get_document", "safe")]


def test_qdrant_search_is_permitted_and_upsert_is_blocked_before_delegate():
    client = FakeClient()
    guard = SearchOnlyQdrant(client)
    assert guard.search("vector") == ["vector"]
    with pytest.raises(EvaluationWriteDenied):
        guard.upsert("collection", [])
    assert guard.blocked_write_attempts == 1
    assert client.calls == [("search", "vector")]


@pytest.mark.parametrize(
    ("guard_type", "mutation"),
    [(ReadOnlyR2, "put"), (ReadOnlyR2, "copy"), (ReadOnlyR2, "delete"), (ReadOnlyCache, "set"), (ReadOnlyCache, "delete")],
)
def test_object_and_cache_mutations_fail_closed(guard_type, mutation):
    guard = guard_type(FakeClient())
    with pytest.raises(EvaluationWriteDenied):
        getattr(guard, mutation)("anything")
    assert guard.blocked_write_attempts == 1


def test_generic_sinks_block_nested_feedback_and_image_mutations():
    sink = MutationDeniedSink("feedback")
    with pytest.raises(EvaluationWriteDenied):
        sink.save("anything")
    with pytest.raises(EvaluationWriteDenied):
        sink("anything")
    assert sink.blocked_write_attempts == 2


def test_factory_shares_one_safe_blocked_write_counter():
    fake = FakeClient()
    reported = []
    guards = EvaluationWriteGuards.create(
        appwrite=fake, qdrant=fake, r2=fake, cache=fake, on_blocked=reported.append
    )
    with pytest.raises(EvaluationWriteDenied):
        guards.appwrite.update_document("x")
    with pytest.raises(EvaluationWriteDenied):
        guards.board_state.register("x")
    with pytest.raises(EvaluationWriteDenied):
        guards.image_generation.generate("x")
    assert guards.counter.count == 3
    assert reported == ["appwrite", "board_state", "image_generation"]
