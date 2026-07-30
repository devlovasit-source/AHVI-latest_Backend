from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import create_style_board_states_collection as migration


def _response(status, data=None, text=""):
    return SimpleNamespace(status_code=status, text=text, json=lambda: deepcopy(data or {}))


def _attribute_rows():
    return [
        {**payload, "type": attr_type}
        for attr_type, payload in migration.ATTRIBUTE_DEFINITIONS
    ]


class FakeAppwrite:
    def __init__(self, *, collection=True, attributes=None, indexes=None):
        self.collection = collection
        self.attributes = deepcopy(_attribute_rows() if attributes is None else attributes)
        self.indexes = deepcopy(list(migration.INDEX_DEFINITIONS) if indexes is None else indexes)
        self.posts = []
        self.deletes = []

    def request(self, method, url, payload=None):
        if method == "DELETE":
            self.deletes.append(url)
            return _response(204)
        if method == "GET" and url.endswith(f"/{migration.COLLECTION_ID}"):
            if not self.collection:
                return _response(404, text="not found")
            return _response(200, {"documentSecurity": False})
        if method == "GET" and url.endswith("/attributes"):
            return _response(200, {"attributes": self.attributes})
        if method == "GET" and url.endswith("/indexes"):
            return _response(200, {"indexes": self.indexes})
        if method == "POST":
            self.posts.append((url, deepcopy(payload)))
            if url == "https://appwrite.test/collections":
                self.collection = True
            elif url.endswith("/attributes/string"):
                self.attributes.append({**payload, "type": "string"})
            elif url.endswith("/attributes/integer"):
                self.attributes.append({**payload, "type": "integer"})
            elif url.endswith("/indexes"):
                self.indexes.append(deepcopy(payload))
            return _response(201)
        raise AssertionError(f"unexpected request: {method} {url}")


@pytest.fixture
def install_fake(monkeypatch):
    def install(fake):
        monkeypatch.setattr(migration, "_base", lambda: "https://appwrite.test/collections")
        monkeypatch.setattr(migration, "_request", fake.request)
        return fake

    return install


def _run_schema_checks():
    migration._ensure_collection()
    migration._ensure_attributes()
    migration._ensure_indexes()


def test_absent_collection_is_created(install_fake):
    fake = install_fake(FakeAppwrite(collection=False))

    migration._ensure_collection()

    assert fake.collection is True
    assert fake.posts[0][1]["collectionId"] == migration.COLLECTION_ID


def test_matching_collection_schema_is_accepted_without_writes(install_fake):
    fake = install_fake(FakeAppwrite())

    _run_schema_checks()

    assert fake.posts == []


def test_missing_attribute_is_created(install_fake):
    rows = [row for row in _attribute_rows() if row["key"] != "createdAtISO"]
    fake = install_fake(FakeAppwrite(attributes=rows))

    migration._ensure_attributes()

    assert fake.posts[-1][1]["key"] == "createdAtISO"


def test_missing_index_is_created(install_fake):
    rows = [row for row in migration.INDEX_DEFINITIONS if row["key"] != "idx_user"]
    fake = install_fake(FakeAppwrite(indexes=rows))

    migration._ensure_indexes()

    assert fake.posts[-1][1]["key"] == "idx_user"


def test_incompatible_attribute_fails_without_recreation(install_fake):
    rows = _attribute_rows()
    next(row for row in rows if row["key"] == "boardId")["size"] = 32
    fake = install_fake(FakeAppwrite(attributes=rows))

    with pytest.raises(RuntimeError, match="attribute boardId schema mismatch"):
        migration._ensure_attributes()

    assert fake.posts == []
    assert fake.deletes == []


def test_incompatible_index_fails_without_recreation(install_fake):
    rows = deepcopy(list(migration.INDEX_DEFINITIONS))
    next(row for row in rows if row["key"] == "idx_board")["orders"] = ["DESC"]
    fake = install_fake(FakeAppwrite(indexes=rows))

    with pytest.raises(RuntimeError, match="index idx_board schema mismatch"):
        migration._ensure_indexes()

    assert fake.posts == []
    assert fake.deletes == []


def test_second_run_performs_no_additional_or_destructive_writes(install_fake):
    fake = install_fake(FakeAppwrite(collection=False, attributes=[], indexes=[]))

    _run_schema_checks()
    first_post_count = len(fake.posts)
    _run_schema_checks()

    assert len(fake.posts) == first_post_count
    assert fake.deletes == []
