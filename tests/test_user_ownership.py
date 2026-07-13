"""Ownership binding: stylist, style-board shuffle and upload endpoints must
bind every operation to the authenticated user (request.state.user).

Contract per endpoint:
- matching body user_id -> proceeds
- missing body user_id -> authenticated user is used
- mismatched body user_id -> HTTP 403
- server-side Appwrite wardrobe reads use the authenticated user id
- inline wardrobe/style assets stay untrusted
- upload storage keys can never target another user's namespace
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from routers import style_boards, stylist, utilities


def _http_request(user_id="user_auth"):
    return SimpleNamespace(
        state=SimpleNamespace(user={"user_id": user_id}),
        url=SimpleNamespace(path="/test"),
    )


def _anon_request():
    return SimpleNamespace(state=SimpleNamespace(), url=SimpleNamespace(path="/test"))


def _item(item_id, name, category):
    return {"id": item_id, "name": name, "category": category, "source": "wardrobe"}


_WARDROBE = [
    _item("dress-1", "Red Dress", "Dresses"),
    _item("sneak-1", "White Sneakers", "Footwear"),
    _item("watch-1", "Gold Watch", "Accessories"),
]


# ---------------------------------------------------------------------------
# stylist: POST /items/{item_id}/style
# ---------------------------------------------------------------------------

def _style_req(**kw):
    return stylist.ItemStyleRequest(
        mode="build_outfit",
        anchor_item=_WARDROBE[0],
        wardrobe=_WARDROBE,
        **kw,
    )


def test_item_style_matching_user_id_succeeds():
    result = stylist.style_wardrobe_item(
        "dress-1", _style_req(user_id="user_auth"), _http_request("user_auth")
    )
    assert result["success"] is True


def test_item_style_missing_user_id_uses_authenticated_user():
    result = stylist.style_wardrobe_item(
        "dress-1", _style_req(), _http_request("user_auth")
    )
    assert result["success"] is True


def test_item_style_mismatched_user_id_403():
    with pytest.raises(HTTPException) as exc:
        stylist.style_wardrobe_item(
            "dress-1", _style_req(user_id="victim"), _http_request("attacker")
        )
    assert exc.value.status_code == 403


def test_item_style_appwrite_read_uses_authenticated_user(monkeypatch):
    captured = {}

    class FakeProxy:
        def list_documents(self, collection, user_id=None, **kw):
            captured[collection] = user_id
            return list(_WARDROBE)

    monkeypatch.setattr(stylist, "AppwriteProxy", FakeProxy)
    request = stylist.ItemStyleRequest(
        user_id="victim_user",  # ignored: mismatches are rejected...
        mode="build_outfit",
        anchor_item=_WARDROBE[0],
        wardrobe=None,  # force the server-side wardrobe read
    )
    with pytest.raises(HTTPException):
        stylist.style_wardrobe_item("dress-1", request, _http_request("user_auth"))
    # ...and with no body user_id the read is bound to the authed user.
    request = stylist.ItemStyleRequest(
        mode="build_outfit", anchor_item=_WARDROBE[0], wardrobe=None
    )
    stylist.style_wardrobe_item("dress-1", request, _http_request("user_auth"))
    assert captured["outfits"] == "user_auth"


# ---------------------------------------------------------------------------
# stylist: POST /pipeline
# ---------------------------------------------------------------------------

def test_pipeline_mismatched_user_id_403():
    request = stylist.OutfitPipelineRequest(user_id="victim", wardrobe=[])
    with pytest.raises(HTTPException) as exc:
        stylist.run_outfit_pipeline(request, _http_request("attacker"))
    assert exc.value.status_code == 403


def test_pipeline_binds_reads_and_flow_to_authenticated_user(monkeypatch):
    captured = {}

    class FakeProxy:
        def list_documents(self, collection, user_id=None, **kw):
            captured["appwrite_user_id"] = user_id
            return []

    def fake_flow(**kw):
        captured["flow_user_id"] = kw.get("user_id")
        return {"success": True, "meta": {}}

    monkeypatch.setattr(stylist, "AppwriteProxy", FakeProxy)
    monkeypatch.setattr(stylist, "build_style_flow_response", fake_flow)
    monkeypatch.setattr(
        stylist.style_dna_engine, "build", lambda payload: {}, raising=False
    )

    request = stylist.OutfitPipelineRequest(user_id="victim")  # mismatched
    with pytest.raises(HTTPException) as exc:
        stylist.run_outfit_pipeline(request, _http_request("user_auth"))
    assert exc.value.status_code == 403

    request = stylist.OutfitPipelineRequest()  # missing -> authed user
    result = stylist.run_outfit_pipeline(request, _http_request("user_auth"))
    assert result["success"] is True
    assert captured["appwrite_user_id"] == "user_auth"
    assert captured["flow_user_id"] == "user_auth"


# ---------------------------------------------------------------------------
# style boards: POST /style-boards/{board_id}/shuffle
# ---------------------------------------------------------------------------

def _shuffle_req(**kw):
    defaults = dict(
        scenario="shuffle_unlocked",
        revision=1,
        shuffle_slots=["top"],
        source_policy="wardrobe",
        wardrobe=_WARDROBE,
    )
    defaults.update(kw)
    return style_boards.BoardShuffleRequest(**defaults)


def test_shuffle_mismatched_user_id_403():
    with pytest.raises(HTTPException) as exc:
        style_boards.shuffle_style_board(
            "board-1", _shuffle_req(user_id="victim"), _http_request("attacker")
        )
    assert exc.value.status_code == 403


def test_shuffle_matching_user_id_proceeds_and_inline_wardrobe_untrusted():
    result = style_boards.shuffle_style_board(
        "board-x", _shuffle_req(user_id="user_auth"), _http_request("user_auth")
    )
    # inline wardrobe path must never be marked trusted
    assert result["wardrobe_source_trusted"] is False


def test_shuffle_appwrite_read_uses_authenticated_user(monkeypatch):
    captured = {}

    class FakeProxy:
        def list_documents(self, collection, user_id=None, **kw):
            captured["user_id"] = user_id
            return []

    monkeypatch.setattr(style_boards, "AppwriteProxy", FakeProxy)
    style_boards.shuffle_style_board(
        "board-y",
        _shuffle_req(user_id="victim_user", wardrobe=None),
        _http_request("victim_user"),
    )
    assert captured["user_id"] == "victim_user"

    style_boards.shuffle_style_board(
        "board-z", _shuffle_req(wardrobe=None), _http_request("user_auth")
    )
    assert captured["user_id"] == "user_auth"


# ---------------------------------------------------------------------------
# uploads
# ---------------------------------------------------------------------------

def test_avatar_upload_mismatched_user_id_403(monkeypatch):
    monkeypatch.setattr(
        utilities.upload_service, "upload_avatar", lambda **kw: "url"
    )
    request = utilities.AvatarUploadRequest(
        user_id="victim", image_base64="aGVsbG8td29ybGQ="
    )
    with pytest.raises(HTTPException) as exc:
        utilities.upload_avatar(request, _http_request("attacker"))
    assert exc.value.status_code == 403


def test_avatar_upload_key_uses_authenticated_user(monkeypatch):
    captured = {}

    def fake_upload(*, user_id, image_base64):
        captured["user_id"] = user_id
        return "https://cdn.test/avatar.png"

    monkeypatch.setattr(utilities.upload_service, "upload_avatar", fake_upload)
    request = utilities.AvatarUploadRequest(image_base64="aGVsbG8td29ybGQ=")
    result = utilities.upload_avatar(request, _http_request("user_auth"))
    assert result == {"avatar_url": "https://cdn.test/avatar.png"}
    assert captured["user_id"] == "user_auth"


def test_avatar_upload_requires_some_user_identity():
    request = utilities.AvatarUploadRequest(image_base64="aGVsbG8td29ybGQ=")
    with pytest.raises(HTTPException) as exc:
        utilities.upload_avatar(request, _anon_request())
    assert exc.value.status_code == 401


def test_wardrobe_upload_namespaces_file_id_by_authenticated_user(monkeypatch):
    captured = {}

    def fake_upload(*, file_id, raw_image_base64, masked_image_base64):
        captured["file_id"] = file_id
        return {"raw_url": "r", "masked_url": "m"}

    monkeypatch.setattr(
        utilities.upload_service, "upload_wardrobe_images", fake_upload
    )
    request = utilities.WardrobeUploadRequest(
        file_id="item123",
        raw_image_base64="aGVsbG8td29ybGQ=",
        masked_image_base64="aGVsbG8td29ybGQ=",
    )
    utilities.upload_wardrobe_images(request, _http_request("user_auth"))
    assert captured["file_id"] == "user_auth-item123"

    # A file_id aimed at another user's namespace still lands in the
    # authenticated user's namespace.
    request = utilities.WardrobeUploadRequest(
        file_id="victim-item123",
        raw_image_base64="aGVsbG8td29ybGQ=",
        masked_image_base64="aGVsbG8td29ybGQ=",
    )
    utilities.upload_wardrobe_images(request, _http_request("user_auth"))
    assert captured["file_id"] == "user_auth-victim-item123"


def test_wardrobe_upload_mismatched_user_id_403(monkeypatch):
    monkeypatch.setattr(
        utilities.upload_service, "upload_wardrobe_images", lambda **kw: {}
    )
    request = utilities.WardrobeUploadRequest(
        file_id="item123",
        user_id="victim",
        raw_image_base64="aGVsbG8td29ybGQ=",
        masked_image_base64="aGVsbG8td29ybGQ=",
    )
    with pytest.raises(HTTPException) as exc:
        utilities.upload_wardrobe_images(request, _http_request("attacker"))
    assert exc.value.status_code == 403


@pytest.mark.parametrize(
    "bad_id", ["../../etc/x", "a/b", "a\\b", "", "x" * 200]
)
def test_wardrobe_upload_rejects_unsafe_file_ids(monkeypatch, bad_id):
    monkeypatch.setattr(
        utilities.upload_service, "upload_wardrobe_images", lambda **kw: {}
    )
    request = utilities.WardrobeUploadRequest(
        file_id=bad_id,
        raw_image_base64="aGVsbG8td29ybGQ=",
        masked_image_base64="aGVsbG8td29ybGQ=",
    )
    with pytest.raises(HTTPException) as exc:
        utilities.upload_wardrobe_images(request, _http_request("user_auth"))
    assert exc.value.status_code == 400
