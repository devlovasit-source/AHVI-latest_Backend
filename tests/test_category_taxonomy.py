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
