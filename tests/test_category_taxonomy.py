from services.category_taxonomy import normalize_category_from_label


def test_tshirt_aliases_normalize_to_tshirt_subcategory():
    for label in (
        "White T-Shirt",
        "Black tshirt",
        "Grey Tee",
        "Crew Neck Top",
        "Round Neck Casual Knit Top",
        "Short Cap Sleeve Tee",
    ):
        assert normalize_category_from_label(label) == ("Tops", "T-Shirt")


def test_tshirt_guard_does_not_rewrite_collared_or_formal_shirts():
    for label in (
        "White Oxford Shirt",
        "Blue Button Down Shirt",
        "Navy Polo Tee",
        "Formal Dress Shirt",
        "Cotton Kurta",
        "Pink Blouse",
        "Black Sweatshirt",
    ):
        category, sub_category = normalize_category_from_label(label)
        assert category != "Item"
        assert sub_category != "T-Shirt"


def test_bottom_aliases_do_not_collapse_to_shorts_or_dresses():
    cases = {
        "Dark Blue Jeans": ("Bottoms", "Jeans"),
        "Distressed Dark Blue Jeans": ("Bottoms", "Jeans"),
        "Grey Trousers": ("Bottoms", "Trousers"),
        "Formal Black Pants": ("Bottoms", "Pants"),
        "Full-length dark blue denim": ("Bottoms", "Jeans"),
        "Black Shorts": ("Bottoms", "Shorts"),
        "Denim Shorts": ("Bottoms", "Shorts"),
    }
    for label, expected in cases.items():
        assert normalize_category_from_label(label) == expected


def test_round_neck_shirt_metadata_can_be_tshirt():
    assert normalize_category_from_label("Neon Yellow Striped Shirt tee round neck") == (
        "Tops",
        "T-Shirt",
    )
