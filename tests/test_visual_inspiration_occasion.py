import pytest
from services.occasion_style_rules import derive_occasion_brief, validate_board_candidate_for_occasion
from services.style_reasoning_engine import _enrich_visual_directions_with_assets

def test_office_outfit_must_not_include_shorts():
    brief = derive_occasion_brief("office outfit")
    item = {"name": "Khaki Shorts", "category": "bottom", "tags": ["shorts"]}
    valid, reason = validate_board_candidate_for_occasion(item, brief)
    assert not valid
    assert "short" in reason

def test_client_meeting_outfit_must_not_include_shorts():
    brief = derive_occasion_brief("client meeting outfit")
    item = {"name": "Chino Shorts", "category": "bottom", "tags": ["shorts"]}
    valid, reason = validate_board_candidate_for_occasion(item, brief)
    assert not valid
    assert "short" in reason

def test_business_professional_outfit_must_not_include_shorts():
    brief = derive_occasion_brief("business professional outfit")
    item = {"name": "Tailored Shorts", "category": "bottom", "tags": ["shorts"]}
    valid, reason = validate_board_candidate_for_occasion(item, brief)
    assert not valid
    assert "short" in reason

def test_smart_casual_office_outfit_must_not_include_shorts():
    brief = derive_occasion_brief("smart casual office outfit")
    item = {"name": "Smart Shorts", "category": "bottom", "tags": ["shorts"]}
    valid, reason = validate_board_candidate_for_occasion(item, brief)
    assert not valid
    assert "short" in reason

def test_beach_vacation_outfit_may_include_shorts():
    brief = derive_occasion_brief("beach vacation outfit")
    item = {"name": "Swim Shorts", "category": "bottom", "tags": ["shorts", "swim"]}
    valid, reason = validate_board_candidate_for_occasion(item, brief)
    assert valid

def test_summer_casual_outfit_may_include_shorts():
    brief = derive_occasion_brief("summer casual outfit")
    item = {"name": "Denim Shorts", "category": "bottom", "tags": ["shorts"]}
    valid, reason = validate_board_candidate_for_occasion(item, brief)
    assert valid

def test_gym_outfit_may_include_shorts():
    brief = derive_occasion_brief("gym outfit")
    item = {"name": "Athletic Shorts", "category": "bottom", "tags": ["shorts", "gym"]}
    valid, reason = validate_board_candidate_for_occasion(item, brief)
    assert valid

def test_enrich_office_outfit_produces_complete_board_after_rejecting_shorts(monkeypatch):
    # Mock _style_asset_rows to return shorts and trousers
    def mock_style_asset_rows(*args, **kwargs):
        return [
            {"asset_id": "1", "status": "active", "name": "Navy Blazer", "category": "top", "role": "outerwear", "image_url": "url1", "gender": "male", "tags": []},
            {"asset_id": "2", "status": "active", "name": "White Oxford Shirt", "category": "top", "role": "top", "image_url": "url2", "gender": "male", "tags": []},
            {"asset_id": "3", "status": "active", "name": "Khaki Shorts", "category": "bottom", "role": "bottom", "image_url": "url3", "gender": "male", "tags": ["shorts"]},
            {"asset_id": "4", "status": "active", "name": "Navy Trousers", "category": "bottom", "role": "bottom", "image_url": "url4", "gender": "male", "tags": ["trousers"]},
            {"asset_id": "5", "status": "active", "name": "Brown Loafers", "category": "footwear", "role": "footwear", "image_url": "url5", "gender": "male", "tags": []},
        ]
    
    import services.style_reasoning_engine as sre
    monkeypatch.setattr(sre, "_style_asset_rows", mock_style_asset_rows)
    
    # Run _enrich_visual_directions_with_assets
    directions = [{
        "title": "Office Look",
        "hero_piece": "Navy Blazer",
        "gender": "male",
        "pieces": ["Navy Blazer", "White Oxford Shirt", "Khaki Shorts", "Brown Loafers"]
    }]
    
    enriched = _enrich_visual_directions_with_assets(
        directions, 
        occasion="office outfit", 
        target_gender="male"
    )
    
    assert len(enriched) == 1
    board_items = enriched[0].get("board_items", [])
    item_names = [bi.get("name") for bi in board_items]
    
    assert "Khaki Shorts" not in item_names
