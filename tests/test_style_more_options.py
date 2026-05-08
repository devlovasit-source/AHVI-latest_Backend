from services.style_flow_service import (
    card_signature,
    finalize_style_cards,
    finalize_style_response_payload,
)


def test_more_options_excludes_already_seen_style_signatures():
    seen_card = {
        "title": "Boardroom Casual",
        "items": [
            {"id": "top-1", "name": "White Shirt", "role": "top", "image_url": "https://x/top1.png"},
            {"id": "bottom-1", "name": "Black Pants", "role": "bottom", "image_url": "https://x/bottom1.png"},
            {"id": "shoe-1", "name": "Black Sneakers", "role": "footwear", "image_url": "https://x/shoe1.png"},
        ],
    }
    fresh_card = {
        "title": "Sharp Daily",
        "items": [
            {"id": "top-2", "name": "Blue Button-Down Shirt", "role": "top", "image_url": "https://x/top2.png"},
            {"id": "bottom-2", "name": "Grey Trousers", "role": "bottom", "image_url": "https://x/bottom2.png"},
            {"id": "shoe-2", "name": "Loafers", "role": "footwear", "image_url": "https://x/shoe2.png"},
        ],
    }

    filtered = finalize_style_cards(
        [seen_card, fresh_card],
        query="office outfit",
        exclude_signatures=[card_signature(seen_card)],
        requested_count=3,
    )

    assert [card_signature(card) for card in filtered] == [card_signature(fresh_card)]


def test_more_options_requested_count_caps_returned_cards():
    cards = [
        {
            "items": [
                {"id": "top-1", "name": "White Shirt", "role": "top", "image_url": "https://x/top1.png"},
                {"id": "bottom-1", "name": "Black Pants", "role": "bottom", "image_url": "https://x/bottom1.png"},
                {"id": "shoe-1", "name": "Black Sneakers", "role": "footwear", "image_url": "https://x/shoe1.png"},
            ]
        },
        {
            "items": [
                {"id": "top-2", "name": "Blue Shirt", "role": "top", "image_url": "https://x/top2.png"},
                {"id": "bottom-2", "name": "Grey Trousers", "role": "bottom", "image_url": "https://x/bottom2.png"},
                {"id": "shoe-2", "name": "Loafers", "role": "footwear", "image_url": "https://x/shoe2.png"},
            ]
        },
        {
            "items": [
                {"id": "top-3", "name": "Off White Shirt", "role": "top", "image_url": "https://x/top3.png"},
                {"id": "bottom-3", "name": "Jeans", "role": "bottom", "image_url": "https://x/bottom3.png"},
                {"id": "shoe-3", "name": "Sneakers", "role": "footwear", "image_url": "https://x/shoe3.png"},
            ]
        },
    ]

    filtered = finalize_style_cards(
        cards,
        query="office outfit",
        requested_count=2,
    )

    assert len(filtered) == 2
    assert all(card_signature(card) for card in filtered)


def test_style_response_contract_uses_single_final_board_list():
    result = {
        "cards": [
            {
                "title": "Signature Combo",
                "items": [
                    {"id": "top-1", "name": "White Shirt", "role": "top", "image_url": "https://x/top1.png"},
                    {"id": "bottom-1", "name": "Black Pants", "role": "bottom", "image_url": "https://x/bottom1.png"},
                    {"id": "shoe-1", "name": "Black Sneakers", "role": "footwear", "image_url": "https://x/shoe1.png"},
                ],
            }
        ],
        "outfits": [],
        "pipeline": {"stages": ["test"]},
    }

    response = finalize_style_response_payload(
        result,
        user_id="u1",
        query="office outfit",
        include_base64=False,
        cache_bypass=True,
    )

    assert response["style_boards"] == response["cards"]
    assert response["data"]["outfits"] == response["cards"]
    assert response["data"]["rendered_boards"] == response["cards"]
    assert response["meta"]["board_count"] == 1
    assert response["meta"]["style_signature"]


def test_accessory_only_change_does_not_create_new_style_board():
    base_core = [
        {"id": "top-1", "name": "White Shirt", "role": "top", "image_url": "https://x/top1.png"},
        {"id": "bottom-1", "name": "Black Pants", "role": "bottom", "image_url": "https://x/bottom1.png"},
        {"id": "shoe-1", "name": "Black Loafers", "role": "footwear", "image_url": "https://x/shoe1.png"},
    ]
    cards = [
        {
            "items": base_core
            + [{"id": "watch-1", "name": "Silver Watch", "role": "accessory", "image_url": "https://x/watch1.png"}],
        },
        {
            "items": base_core
            + [{"id": "watch-2", "name": "Gold Watch", "role": "accessory", "image_url": "https://x/watch2.png"}],
        },
    ]

    filtered = finalize_style_cards(cards, query="office outfit")

    assert len(filtered) == 1
    assert filtered[0]["_style_core_signature"] == "top-1|bottom-1|shoe-1"
