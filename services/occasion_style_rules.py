import logging
from typing import Any, Dict

logger = logging.getLogger("ahvi.occasion_rules")

_OFFICE_OCCASIONS = {
    "office", "business", "business_casual", "business casual",
    "client meeting", "client_meeting", "boardroom", "corporate",
    "professional", "smart casual", "smart_casual", "executive",
    "contemporary classic office"
}

_RELAXED_OCCASIONS = {
    "beach", "vacation", "resort", "lounge", "casual", "weekend", "gym",
    "workout", "athletic", "sports", "running", "summer casual"
}

def derive_occasion_brief(occasion_text: str, is_formal: bool = False) -> Dict[str, Any]:
    """Derive occasion rules based on prompt text."""
    occasion = (occasion_text or "").strip().lower()
    
    is_office = any(o in occasion for o in _OFFICE_OCCASIONS)
    is_relaxed = any(o in occasion for o in _RELAXED_OCCASIONS)
    
    blocked_tokens = set()
    blocked_families = set()
    
    if is_office:
        # Professional/office/client meeting/business prompts
        blocked_tokens.update([
            "shorts", "short", "swimshorts",
            "boxer", "underwear", "sleepwear", "loungewear", "lounge", "pyjama", "pajama",
            "swim", "swimwear", "trunk",
            "distressed denim", "distressed", "ripped",
            "hoodie", "hoodies",
            "slipper", "slippers", "sandal", "sandals", "slide", "slider", "flip flop", "flip-flop",
            "running shoe", "athletic shoe", "gym shoe", "trainer", "running", "athletic", "gym",
            "party shirt", "festive shirt", "flashy"
        ])
        blocked_families.update(["shorts", "swimwear", "loungewear", "gymwear"])
        
    brief = {
        "occasion": occasion,
        "formality": "high" if is_office or is_formal else "low",
        "archetype": "professional" if is_office else "casual",
        "required_slots": ["top", "bottom", "footwear"],
        "allowed_roles": ["top", "bottom", "footwear", "accessory", "outerwear", "hero"],
        "blocked_families": list(blocked_families),
        "blocked_tokens": list(blocked_tokens),
        "footwear_policy": "smart" if is_office else "any",
        "bottom_policy": "trousers_only" if is_office else "any"
    }
    
    logger.info(f"AHVI_VISUAL_BRIEF_NORMALIZED occasion='{occasion}' brief={brief}")
    return brief

def validate_board_candidate_for_occasion(candidate: Dict[str, Any], occasion_brief: Dict[str, Any], board_title: str = "") -> tuple[bool, str]:
    """Validate a single candidate item before adding it to a board."""
    if not isinstance(candidate, dict):
        return False, "invalid_item"
        
    occasion = occasion_brief.get("occasion", "")
    item_id = candidate.get("id") or candidate.get("item_id") or ""
    item_name = candidate.get("name") or candidate.get("title") or ""
    role = candidate.get("role") or candidate.get("category") or ""
        
    blocked_tokens = occasion_brief.get("blocked_tokens", [])
    if not blocked_tokens:
        logger.info(f"AHVI_VISUAL_CANDIDATE_ACCEPTED occasion='{occasion}' item_name='{item_name}'")
        return True, ""
        
    # Build text blob
    parts = []
    for key in ["name", "title", "category", "sub_category", "subcategory", "type", "tags", "style_tags"]:
        val = candidate.get(key)
        if isinstance(val, list):
            parts.extend([str(v) for v in val])
        elif val:
            parts.append(str(val))
            
    blob = " " + " ".join(parts).lower() + " "
    
    for token in blocked_tokens:
        # Padded check
        if f" {token} " in blob or f" {token}s " in blob or f" {token}es " in blob:
            reason = f"blocked_token:{token}"
            logger.info(
                f"AHVI_VISUAL_CANDIDATE_REJECTED occasion='{occasion}' title='{board_title}' "
                f"item_id='{item_id}' item_name='{item_name}' role='{role}' reason='{reason}'"
            )
            return False, reason
            
        # Direct substring check
        if token in ["distressed", "ripped", "swim", "lounge"]:
            if token in blob:
                reason = f"blocked_token:{token}"
                logger.info(
                    f"AHVI_VISUAL_CANDIDATE_REJECTED occasion='{occasion}' title='{board_title}' "
                    f"item_id='{item_id}' item_name='{item_name}' role='{role}' reason='{reason}'"
                )
                return False, reason
                
    logger.info(f"AHVI_VISUAL_CANDIDATE_ACCEPTED occasion='{occasion}' item_name='{item_name}'")
    return True, ""
