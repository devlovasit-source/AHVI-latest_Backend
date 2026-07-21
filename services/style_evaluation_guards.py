"""Fail-closed persistence boundaries for internal Style evaluation.

These wrappers are deliberately small and transport-agnostic.  They are used
by the internal evaluator to retain the production read/composition path while
making an accidental mutation fail *before* it reaches a network client.  Do
not put request data in exceptions or logs emitted by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, Optional


class EvaluationWriteDenied(RuntimeError):
    """A write was attempted from a write-denied evaluation execution."""

    code = "STYLE_EVALUATION_WRITE_DENIED"

    def __init__(self, boundary: str) -> None:
        # Boundary is a fixed internal label, never a URL/payload/client value.
        self.boundary = _safe_boundary(boundary)
        super().__init__(self.code)


def _safe_boundary(boundary: str) -> str:
    value = "".join(ch for ch in str(boundary) if ch.isalnum() or ch in "._-")
    return value[:64] or "unknown"


@dataclass
class BlockedWriteCounter:
    """Safe, injectable accounting for a blocked mutation attempt."""

    count: int = 0
    on_blocked: Optional[Callable[[str], None]] = None

    def record(self, boundary: str) -> None:
        self.count += 1
        if self.on_blocked is not None:
            self.on_blocked(_safe_boundary(boundary))


class _GuardedClient:
    _read_methods: FrozenSet[str] = frozenset()

    def __init__(self, client: Any, *, boundary: str, counter: Optional[BlockedWriteCounter] = None) -> None:
        self._client = client
        self._boundary = _safe_boundary(boundary)
        self._counter = counter or BlockedWriteCounter()

    @property
    def blocked_write_attempts(self) -> int:
        return self._counter.count

    def _deny(self) -> None:
        self._counter.record(self._boundary)
        raise EvaluationWriteDenied(self._boundary)

    def __getattr__(self, name: str) -> Any:
        # Explicit allow-lists are intentional: a new SDK method must be
        # consciously classified before evaluation can use it.
        if name in self._read_methods:
            return getattr(self._client, name)
        return self._denied_method

    def _denied_method(self, *_args: Any, **_kwargs: Any) -> None:
        self._deny()


class ReadOnlyAppwrite(_GuardedClient):
    """Appwrite facade allowing document/schema reads and denying mutations."""

    _read_methods = frozenset(
        {
            "get_document",
            "list_documents",
            "get_collection",
            "list_collections",
            "get_attribute",
            "list_attributes",
            "get_file",
            "get_file_download",
            "get_file_preview",
            "get_file_view",
        }
    )

    def __init__(self, client: Any, *, counter: Optional[BlockedWriteCounter] = None) -> None:
        super().__init__(client, boundary="appwrite", counter=counter)


class SearchOnlyQdrant(_GuardedClient):
    """Qdrant facade allowing retrieval/search, never vector mutation."""

    _read_methods = frozenset({"search", "query_points", "scroll", "retrieve", "count", "get_collection"})

    def __init__(self, client: Any, *, counter: Optional[BlockedWriteCounter] = None) -> None:
        super().__init__(client, boundary="qdrant", counter=counter)


class ReadOnlyR2(_GuardedClient):
    """Object-storage facade. Downloads may be used; object mutations cannot."""

    _read_methods = frozenset({"get", "head", "download", "exists", "list"})

    def __init__(self, client: Any, *, counter: Optional[BlockedWriteCounter] = None) -> None:
        super().__init__(client, boundary="r2", counter=counter)


class ReadOnlyCache(_GuardedClient):
    """Cache facade permitting lookups while denying any cache mutation."""

    _read_methods = frozenset({"get", "get_many", "exists", "ttl", "keys"})

    def __init__(self, client: Any, *, counter: Optional[BlockedWriteCounter] = None) -> None:
        super().__init__(client, boundary="cache", counter=counter)


class MutationDeniedSink:
    """Sink for board/feedback/save/wear/plan/image operations in evaluation."""

    def __init__(self, boundary: str, *, counter: Optional[BlockedWriteCounter] = None) -> None:
        self._boundary = _safe_boundary(boundary)
        self._counter = counter or BlockedWriteCounter()

    @property
    def blocked_write_attempts(self) -> int:
        return self._counter.count

    def __call__(self, *_args: Any, **_kwargs: Any) -> None:
        self._counter.record(self._boundary)
        raise EvaluationWriteDenied(self._boundary)

    def __getattr__(self, _name: str) -> Callable[..., None]:
        return self


@dataclass(frozen=True)
class EvaluationWriteGuards:
    """One shared counter plus all side-effect boundaries for an evaluation."""

    counter: BlockedWriteCounter
    appwrite: ReadOnlyAppwrite
    qdrant: SearchOnlyQdrant
    r2: ReadOnlyR2
    cache: ReadOnlyCache
    board_state: MutationDeniedSink
    feedback: MutationDeniedSink
    image_generation: MutationDeniedSink

    @classmethod
    def create(
        cls,
        *,
        appwrite: Any,
        qdrant: Any,
        r2: Any,
        cache: Any,
        on_blocked: Optional[Callable[[str], None]] = None,
    ) -> "EvaluationWriteGuards":
        counter = BlockedWriteCounter(on_blocked=on_blocked)
        return cls(
            counter=counter,
            appwrite=ReadOnlyAppwrite(appwrite, counter=counter),
            qdrant=SearchOnlyQdrant(qdrant, counter=counter),
            r2=ReadOnlyR2(r2, counter=counter),
            cache=ReadOnlyCache(cache, counter=counter),
            board_state=MutationDeniedSink("board_state", counter=counter),
            feedback=MutationDeniedSink("feedback", counter=counter),
            image_generation=MutationDeniedSink("image_generation", counter=counter),
        )
