# utils/wardrobe_parser.py

import re
from typing import List, Dict, Any


def extract_and_clean_response(llama_text: str, wardrobe: List[Dict[str, Any]]) -> dict:
    """
    Elite parser:
    - Extracts CHIPS, PACK_LIST, STYLE_BOARD
    - Validates STYLE_BOARD IDs against wardrobe
    - Cleans hallucinated IDs from text
    - Returns UI-ready structure
    """

    response_data = {
        "cleaned_text": llama_text or "",
        "chips": [],
        "pack_tag": "",
        "board_tag": "",
        "board_ids": []  # 🔥 NEW (important for UI)
    }

    text = response_data["cleaned_text"]

    # =========================
    # 1. CHIPS
    # =========================
    chip_match = re.search(r'\[CHIPS?:\s*(.*?)\]', text, re.IGNORECASE)
    if chip_match:
        chips = chip_match.group(1)
        response_data["chips"] = [
            c.strip() for c in chips.split(",") if c.strip()
        ]

    text = re.sub(r'\[CHIPS?:.*?\]', '', text, flags=re.IGNORECASE)

    # =========================
    # 2. PACK LIST
    # =========================
    pack_match = re.search(r'\[?PACK_LIST:\s*(.*?)(?:\]|\n|$)', text, re.IGNORECASE)
    if pack_match:
        raw_pack = pack_match.group(1).strip()
        if raw_pack:
            response_data["pack_tag"] = f"[PACK_LIST: {raw_pack}]"

    text = re.sub(r'\[?PACK_LIST:.*?(\]|\n|$)', '', text, flags=re.IGNORECASE)

    # =========================
    # 3. STYLE BOARD
    # =========================
    board_match = re.search(r'\[?STYLE_BOARD:\s*(.*?)(?:\]|\n|$)', text, re.IGNORECASE)

    if board_match:
        raw_items = board_match.group(1).strip()

        if raw_items:
            # split ids
            ids = [i.strip() for i in raw_items.split(",") if i.strip()]

            # 🔥 VALIDATE AGAINST WARDROBE
            valid_ids = []
            wardrobe_ids = {
                str(item.get("$id") or item.get("id"))
                for item in wardrobe
            }

            for i in ids:
                if i in wardrobe_ids:
                    valid_ids.append(i)

            if valid_ids:
                response_data["board_ids"] = valid_ids
                response_data["board_tag"] = f"[STYLE_BOARD: {', '.join(valid_ids)}]"

    text = re.sub(r'\[?STYLE_BOARD:.*?(\]|\n|$)', '', text, flags=re.IGNORECASE)

    # =========================
    # 4. CLEAN TEXT (IMPORTANT)
    # =========================

    # remove wardrobe ids
    for item in wardrobe:
        item_id = str(item.get("$id") or item.get("id", ""))
        if item_id:
            text = text.replace(item_id, "")

    # remove generic junk tokens
    text = re.sub(r'\b(id\d+|item\d+|items?|ids?)\b', '', text, flags=re.IGNORECASE)

    # remove empty brackets / artifacts
    text = re.sub(r'\(\s*[,\s]*\)', '', text)

    # normalize spaces
    text = re.sub(r'\s{2,}', ' ', text).strip()

    response_data["cleaned_text"] = text

    return response_data
