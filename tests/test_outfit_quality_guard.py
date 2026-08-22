from brain.engines.outfit_quality_guard import guard_outfit, filter_and_guard_outfits


def test_yellow_athletic_shoe_penalized_for_date_night():
    outfit = {
        "top": {"name": "Emerald Shirt", "category": "shirt", "color": "emerald"},
        "bottom": {"name": "Beige Tailored Trousers", "category": "trousers", "color": "beige"},
        "footwear": {"name": "Yellow Running Shoes", "category": "running shoes", "color": "yellow"},
        "score": 80,
    }

    allowed, penalty, reasons, fixed = guard_outfit(
        outfit,
        user_profile={"gender": "male"},
        intent="date night",
        query="what should I wear tonight",
    )

    assert allowed is False or penalty <= -70
    assert reasons


def test_yellow_shoe_allowed_when_bold_streetwear_requested():
    outfit = {
        "top": {"name": "Emerald Shirt", "category": "shirt", "color": "emerald"},
        "bottom": {"name": "Black Jeans", "category": "jeans", "color": "black"},
        "footwear": {"name": "Yellow Sneakers", "category": "sneakers", "color": "yellow"},
        "score": 80,
    }

    allowed, penalty, reasons, fixed = guard_outfit(
        outfit,
        user_profile={"gender": "male"},
        intent="streetwear",
        query="bold sporty streetwear look",
    )

    assert allowed is True


def test_male_saree_blocked():
    outfit = {
        "top": {"name": "Green Saree", "category": "saree", "color": "green"},
        "bottom": {"name": "Jeans", "category": "jeans", "color": "blue"},
        "footwear": {"name": "White Sneakers", "category": "sneakers", "color": "white"},
        "score": 80,
    }

    allowed, penalty, reasons, fixed = guard_outfit(
        outfit,
        user_profile={"gender": "male"},
        intent="casual",
        query="suggest outfit",
    )

    assert allowed is False


def test_accessories_one_per_type():
    outfits = [
        {
            "top": {"name": "White Shirt", "category": "shirt"},
            "bottom": {"name": "Blue Jeans", "category": "jeans"},
            "footwear": {"name": "White Sneakers", "category": "sneakers"},
            "accessories": [
                {"name": "Gold Watch", "category": "watch"},
                {"name": "Black Watch", "category": "watch"},
                {"name": "Bracelet", "category": "bracelet"},
            ],
            "score": 50,
        }
    ]

    guarded = filter_and_guard_outfits(
        outfits,
        user_profile={"gender": "male"},
        intent="smart casual",
        query="",
    )

    assert len(guarded) == 1
    assert len(guarded[0]["accessories"]) == 2


def test_boardroom_casual_rejects_party_shirt_and_red_loafers():
    board = {
        "title": "Boardroom Casual",
        "items": [
            {"name": "Short Sleeve Navy Blue Embroidered Party Shirt", "category": "top"},
            {"name": "Black Pants", "category": "bottom"},
            {"name": "Red Loafers", "category": "footwear"},
            {"name": "Gold Watch", "category": "accessory"},
        ],
    }

    rejected, reason = reject_board_for_occasion(board, "boardroom casual")

    assert rejected is True
    assert reason.startswith("office_forbidden_")


def test_date_night_rejects_shorts_even_with_polished_top():
    board = {
        "title": "Date Night Edit",
        "items": [
            {"name": "Mint Green Shirt", "category": "top"},
            {"name": "Casual Shorts", "category": "bottom"},
            {"name": "Loafers", "category": "footwear"},
        ],
    }

    rejected, reason = reject_board_for_occasion(board, "date night")

    assert rejected is True
    assert reason.startswith("date_forbidden_")
from brain.engines.outfit_quality_guard import reject_board_for_occasion
