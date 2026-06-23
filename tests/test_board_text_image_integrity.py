"""Board text/image integrity: pieces text + rationale must describe the items
ACTUALLY on the board (board_items), never the LLM's ideal piece names."""
from services.style_reasoning_engine import (
    board_text_image_mismatch,
    canonical_piece_names,
)


def _bi(name, role):
    return {"name": name, "role": role, "image_url": f"https://x/{role}.png"}


BOARD = [
    _bi("Blue Polo", "top"),
    _bi("Cream Shorts", "bottom"),
    _bi("Brown Formal Shoes", "footwear"),
]


def test_canonical_names_from_board_items():
    assert canonical_piece_names(BOARD) == ["Blue Polo", "Cream Shorts", "Brown Formal Shoes"]


def test_A_shorts_shown_but_text_says_trouser_is_mismatch():
    pieces_text = ["Fine-Gauge Knit Polo", "Tailored Khaki Trouser", "Brown Formal Shoes"]
    assert board_text_image_mismatch(BOARD, pieces_text) is True
    # repair = canonical names (the real shorts, not trouser)
    assert "Cream Shorts" in canonical_piece_names(BOARD)
    assert "Tailored Khaki Trouser" not in canonical_piece_names(BOARD)


def test_B_formal_shoes_shown_but_text_says_sneakers_is_mismatch():
    pieces_text = ["Blue Polo", "Cream Shorts", "Clean White Leather Sneakers"]
    assert board_text_image_mismatch(BOARD, pieces_text) is True


def test_matching_text_is_no_mismatch():
    pieces_text = ["Blue Polo", "Cream Shorts", "Brown Formal Shoes"]
    assert board_text_image_mismatch(BOARD, pieces_text) is False


def test_D_owned_items_not_appended_to_board_items():
    # canonical names come ONLY from board_items; owned_items are never merged.
    owned = [_bi("Old Hoodie", "top"), _bi("Gym Shorts", "bottom")]
    names = canonical_piece_names(BOARD)  # owned not passed -> not present
    assert "Old Hoodie" not in names
    assert "Gym Shorts" not in names
    assert names == ["Blue Polo", "Cream Shorts", "Brown Formal Shoes"]


def test_E_rationale_source_is_final_names_only():
    # Anything not in canonical_piece_names is a mismatch -> would be repaired.
    rationale_mentions = ["Blue Polo", "Tailored Khaki Trouser"]
    assert board_text_image_mismatch(BOARD, rationale_mentions) is True


def test_empty_board_is_safe():
    assert canonical_piece_names([]) == []
    assert board_text_image_mismatch([], ["anything"]) is False
