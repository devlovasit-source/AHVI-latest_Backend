from routers import wardrobe_capture as wc
from services import gemini_multi_garment_detector as gmg
from services import wardrobe_taxonomy as wt
import services.hybrid_detection_service as hybrid


def _guarded_cap(name: str, *, color: str = "white", context: str = "baseball cap"):
    item = {
        "name": name,
        "category": "Tops",
        "sub_category": "Top",
        "subcategory": "Top",
        "color_name": color,
        "confidence": 0.92,
    }
    return wc._apply_headwear_ocr_guard(item, context_text=context)


def test_watsonx_cap_ocr_contamination_rewrites_to_accessory_cap():
    out = _guarded_cap("Watsonx Socks", color="white", context="white baseball cap")

    assert out["category"] == "Accessories"
    assert out["sub_category"] == "Cap"
    assert out["subcategory"] == "Cap"
    assert out["name"] in {"White Baseball Cap", "Watsonx Cap", "White Cap"}
    assert "sock" not in out["name"].lower()
    assert "top" not in out["name"].lower()


def test_nike_cap_does_not_remain_socks_or_tops():
    out = _guarded_cap("Nike Socks", color="", context="black cap")

    assert out["category"] == "Accessories"
    assert out["sub_category"] == "Cap"
    assert out["name"] == "Nike Cap"


def test_adidas_cap_does_not_remain_top_or_tops():
    out = _guarded_cap("Adidas Top", color="", context="dad cap")

    assert out["category"] == "Accessories"
    assert out["sub_category"] == "Cap"
    assert out["name"] == "Adidas Cap"


def test_logo_only_weak_top_without_cap_signal_goes_to_review():
    out = wc._apply_headwear_ocr_guard(
        {
            "name": "IBM Top",
            "category": "Tops",
            "sub_category": "Top",
            "confidence": 0.31,
        }
    )

    assert out["category"] == "Needs Review"
    assert out["sub_category"] == "Needs Review"


def test_black_baseball_cap_taxonomy_is_accessories_cap():
    category, sub = wt.normalize(name="Black Baseball Cap", category="Tops", sub_category="Top")

    assert category == "Accessories"
    assert sub == "Cap"


def test_white_dad_cap_taxonomy_is_accessories_cap():
    category, sub = wt.normalize(name="White Dad Cap", category="Tops", sub_category="Top")

    assert category == "Accessories"
    assert sub == "Cap"


def test_visor_taxonomy_is_accessory_not_top():
    category, sub = wt.normalize(name="White Visor", category="Tops", sub_category="Top")

    assert category == "Accessories"
    assert sub in {"Cap", "Visor"}


def test_beanie_taxonomy_is_accessory_beanie():
    category, sub = wt.normalize(name="Grey Beanie", category="Tops", sub_category="Top")

    assert category == "Accessories"
    assert sub == "Beanie"


def test_leather_belt_wrist_watch_and_tote_bag_never_become_tops():
    assert wt.normalize(name="Leather Belt", category="Accessories", sub_category="Belt") == (
        "Accessories",
        "Belt",
    )
    assert wt.normalize(name="Wrist Watch", category="Accessories", sub_category="Watch") == (
        "Accessories",
        "Watch",
    )
    tote_cat, tote_sub = wt.normalize(name="Canvas Tote Bag", category="Bags", sub_category="Tote Bag")
    assert tote_cat in {"Bags", "Accessories"}
    assert tote_sub in {"Tote Bag", "Bag"}
    assert tote_cat != "Tops"


def test_detector_prompts_include_headwear_terms():
    gemini_supported = " ".join(gmg.SUPPORTED_ITEMS).lower()
    hybrid_prompt = hybrid.TEXT_PROMPT.lower()
    for term in ("cap", "hat", "headwear", "baseball cap", "snapback", "dad cap", "visor", "beanie"):
        assert term in gemini_supported
        assert term in hybrid_prompt
