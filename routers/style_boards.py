"""Style board endpoints: lock-aware shuffle.

POST /api/style-boards/{board_id}/shuffle
Structured contract (NOT free text through chat): the client sends the board
revision, locked items with exact image URLs + positions, the slots to
shuffle, exclusions and the source policy.  Locked items are hard-guaranteed
to survive (see services.style_item_contract.assert_fixed_items_preserved).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.appwrite_proxy import AppwriteProxy
from services import style_board_shuffle_service

router = APIRouter()
logger = logging.getLogger("ahvi.style_boards")


class BoardPosition(BaseModel):
    x: float = 0.0
    y: float = 0.0
    width: float = 0.3
    height: float = 0.3
    z: int = 1
    rotation: float = 0.0


class LockedBoardItem(BaseModel):
    item_id: str
    slot: str = ""
    role: str = ""
    source: str = ""
    image_url: str = ""
    name: str = ""
    position: Optional[BoardPosition] = None


class BoardShuffleRequest(BaseModel):
    user_id: str
    scenario: str = "shuffle_unlocked"
    revision: int = 1
    occasion: Optional[str] = None
    style_direction: Optional[str] = None
    locked_items: List[Dict[str, Any]] = Field(default_factory=list)
    shuffle_slots: List[str] = Field(default_factory=list)
    exclude_item_ids: List[str] = Field(default_factory=list)
    source_policy: Union[str, Dict[str, Any]] = "inherit"
    board_items: List[Dict[str, Any]] = Field(default_factory=list)
    wardrobe: Any = None  # optional inline wardrobe (tests / offline)
    style_assets: Any = None  # optional inline style assets (tests / offline)


def _resolve_wardrobe(request: BoardShuffleRequest) -> "tuple[List[Dict[str, Any]], bool]":
    """Same resolution as routers.stylist._resolve_wardrobe, plus provenance
    stamping: documents listed from the user's own wardrobe collection ARE
    wardrobe items, so a missing source is stamped explicitly (the contract
    treats unknown != wardrobe on purpose)."""
    rows: List[Dict[str, Any]]
    if isinstance(request.wardrobe, list):
        rows = [dict(i) for i in request.wardrobe if isinstance(i, dict)]
        return rows, False
    else:
        try:
            docs = AppwriteProxy().list_documents("outfits", user_id=request.user_id) or []
            rows = [i for i in docs if isinstance(i, dict)]
        except Exception:
            rows = []
    return [i if i.get("source") else {**i, "source": "wardrobe"} for i in rows], True


def _style_asset_provider() -> List[Dict[str, Any]]:
    """Same curated style-asset source the initial Style This generation uses
    (routers.stylist._resolve_style_assets): the style_assets collection with
    explicit provenance stamping."""
    try:
        rows = AppwriteProxy().list_documents("style_assets") or []
    except Exception:
        return []
    return [
        {**i, "source": i.get("source") or "style_asset"}
        for i in rows if isinstance(i, dict)
    ]


@router.post("/style-boards/{board_id}/shuffle")
def shuffle_style_board(board_id: str, request: BoardShuffleRequest) -> Dict[str, Any]:
    wardrobe, wardrobe_source_trusted = _resolve_wardrobe(request)
    locked = [dict(item) for item in request.locked_items]
    inline_assets = (
        [dict(i) for i in request.style_assets if isinstance(i, dict)]
        if isinstance(request.style_assets, list) else None
    )

    result = style_board_shuffle_service.shuffle_board(
        board_id=board_id,
        revision=request.revision,
        locked_items=locked,
        shuffle_slots=request.shuffle_slots,
        exclude_item_ids=request.exclude_item_ids,
        occasion=request.occasion,
        source_policy=request.source_policy,
        wardrobe=wardrobe,
        style_assets=inline_assets,
        style_asset_provider=None if inline_assets is not None else _style_asset_provider,
        context={
            "board_items": request.board_items,
            "style_direction": request.style_direction,
        },
    )
    result["locked_items"] = locked
    result["shuffle_slots"] = list(request.shuffle_slots)
    result["exclude_item_ids"] = list(request.exclude_item_ids)
    # The service reports the RESOLVED board policy; keep the raw request
    # value separately for observability.
    result["requested_source_policy"] = request.source_policy
    result.setdefault("source_policy", None)
    result["wardrobe_source_trusted"] = wardrobe_source_trusted
    result.setdefault("board_id", board_id)
    result.setdefault("revision", request.revision)
    result.setdefault("board_items", [])
    result.setdefault("changed_slots", [])
    result.setdefault("locked_items_preserved", False)
    result.setdefault("previous_revision", request.revision)
    logger.info(
        "style_boards.shuffle board=%s rev=%s success=%s changed=%s",
        board_id, request.revision, result.get("success"),
        result.get("changed_slots"),
    )
    return result
