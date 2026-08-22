"""Tests for the H&M docx parser and Meghna normalizer pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import convert_hm_docx_to_style_assets as hm
from scripts import normalize_meghna_style_assets as meghna


# ---------- H&M parser ----------

def _hm_paragraphs():
    return [
        "MENS",
        "T-SHIRTS",
        "https://image.hm.com/assets/hm/t1.jpg?imwidth=2160",
        "https://image.hm.com/assets/hm/t2.jpg?imwidth=2160",
        "SHIRTS",
        "https://image.hm.com/assets/hm/s1.jpg",
        "JEANS",
        "https://image.hm.com/assets/hm/j1.jpg",
        "SHORTS",
        "https://image.hm.com/assets/hm/sh1.jpg",
        "JACKETS&COATS",
        "https://image.hm.com/assets/hm/jk1.jpg",
        "Hoodies & Sweatshirts",
        "https://image.hm.com/assets/hm/h1.jpg",
        "Sleepwear & Loungewear",
        "https://image.hm.com/assets/hm/sl1.jpg",
        "FOOTWARE",
        "1.SHOES",
        "https://image.hm.com/assets/hm/sho1.jpg",
        "2.FLIP FLOPS",
        "https://image.hm.com/assets/hm/ff1.jpg",
        "ACCESSORIES",
        "Bracelets:",
        "https://image.hm.com/assets/hm/br1.jpg",
        "Necklaces:",
        "https://image.hm.com/assets/hm/n1.jpg",
        "3.Rings:",
        "https://image.hm.com/assets/hm/r1.jpg",
        "Earrings:",
        "https://image.hm.com/assets/hm/e1.jpg",
        "Sunglasses:",
        "https://image.hm.com/assets/hm/sg1.jpg",
        "Belts:",
        "https://image.hm.com/assets/hm/b1.jpg",
        "Hats:",
        "https://image.hm.com/assets/hm/ht1.jpg",
    ]


def test_hm_parser_groups_urls_by_category():
    rows = hm.parse_document(_hm_paragraphs())
    by_sub = {row["subcategory"]: row for row in rows}
    assert ("top", "t_shirt") == (by_sub["t_shirt"]["category"], by_sub["t_shirt"]["subcategory"])
    assert ("top", "shirt") == (by_sub["shirt"]["category"], by_sub["shirt"]["subcategory"])
    assert ("bottom", "jeans") == (by_sub["jeans"]["category"], by_sub["jeans"]["subcategory"])
    assert ("outerwear", "jacket") == (by_sub["jacket"]["category"], by_sub["jacket"]["subcategory"])
    assert ("top", "hoodie_sweatshirt") == (
        by_sub["hoodie_sweatshirt"]["category"],
        by_sub["hoodie_sweatshirt"]["subcategory"],
    )
    assert ("footwear", "shoes") == (by_sub["shoes"]["category"], by_sub["shoes"]["subcategory"])
    assert ("footwear", "flip_flops") == (by_sub["flip_flops"]["category"], by_sub["flip_flops"]["subcategory"])
    for sub in ("bracelet", "necklace", "ring", "sunglasses", "belt", "hat"):
        assert by_sub[sub]["category"] == "accessory"


def test_hm_parser_handles_multiple_urls_in_one_paragraph():
    paras = ["T-SHIRTS", "https://x/a.jpg https://x/b.jpg\nhttps://x/c.jpg"]
    rows = hm.parse_document(paras)
    assert len(rows) == 3
    assert {row["image_url"] for row in rows} == {"https://x/a.jpg", "https://x/b.jpg", "https://x/c.jpg"}


def test_hm_parser_marks_sleepwear_inactive():
    rows = hm.parse_document(_hm_paragraphs())
    sleep = [r for r in rows if r["subcategory"] == "sleepwear"]
    assert sleep and all(r["status"] == "inactive" for r in sleep)


def test_hm_parser_excludes_earrings_from_mens_seed():
    rows = hm.parse_document(_hm_paragraphs())
    assert not any(r["subcategory"] == "earrings" for r in rows)


def test_hm_parser_does_not_tag_shorts_for_office_or_coffee_date():
    rows = hm.parse_document(_hm_paragraphs())
    shorts = next(r for r in rows if r["subcategory"] == "shorts")
    flip = next(r for r in rows if r["subcategory"] == "flip_flops")
    for row in (shorts, flip):
        for off in row["occasions"]:
            assert off not in {"startup_office", "coffee_date", "wedding", "funeral", "client_meeting"}


def test_hm_parser_gender_defaults():
    rows = hm.parse_document(_hm_paragraphs())
    necklace = next(r for r in rows if r["subcategory"] == "necklace")
    bracelet = next(r for r in rows if r["subcategory"] == "bracelet")
    assert necklace["gender"] == "unisex"
    assert bracelet["gender"] == "male"


def test_hm_parser_stable_asset_ids():
    rows1 = hm.parse_document(_hm_paragraphs())
    rows2 = hm.parse_document(_hm_paragraphs())
    assert [r["asset_id"] for r in rows1] == [r["asset_id"] for r in rows2]


# ---------- Meghna normalizer ----------

def test_meghna_handles_double_brace_typo(tmp_path: Path):
    bad = tmp_path / "asset.json"
    bad.write_text(
        '{{"category": "tops", "assets": [{"asset_id": "white_tee", "name": "White Tee", '
        '"subcategory": "tshirt", "gender_support": ["male"], "imageUrl": "https://x/wt.png"}]}',
        encoding="utf-8",
    )
    rows, report = meghna.normalize(bad)
    assert report["summary"]["total"] == 1
    assert rows[0]["image_url"] == "https://x/wt.png"
    assert rows[0]["category"] == "top"
    assert rows[0]["subcategory"] == "t_shirt"
    assert rows[0]["gender"] == "male"


def test_meghna_normalizes_field_aliases(tmp_path: Path):
    payload = [
        {"id": "bag1", "name": "Crossbody Bag", "category": "bags", "gender": "women", "image_url": "https://x/b.png"},
        {"$id": "n1", "name": "Pearl Necklace", "categoryName": "necklaces", "men": False, "url": "https://x/n.png"},
    ]
    p = tmp_path / "x.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    rows, _ = meghna.normalize(p)
    by_name = {r["name"]: r for r in rows}
    assert by_name["Crossbody Bag"]["gender"] == "female"
    assert by_name["Crossbody Bag"]["category"] == "accessory"
    assert by_name["Crossbody Bag"]["subcategory"] == "bag"
    assert by_name["Pearl Necklace"]["subcategory"] == "necklace"


def test_meghna_preserves_asset_path_when_no_url(tmp_path: Path):
    payload = {
        "category": "bottoms",
        "assets": [
            {"asset_id": "bj", "name": "Blue Jeans", "subcategory": "jeans", "gender_support": ["male"], "asset_path": "mens/bottoms/bluejeans.jpg"}
        ],
    }
    p = tmp_path / "b.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    rows, report = meghna.normalize(p)
    assert rows and rows[0]["asset_path"] == "mens/bottoms/bluejeans.jpg"
    assert rows[0]["image_url"] == ""
    assert report["summary"]["missing_image_url"] == 1


def test_meghna_rejects_missing_required(tmp_path: Path):
    payload = {
        "assets": [
            {"asset_id": "nameless", "category": "tops", "gender_support": ["male"], "image_url": "https://x/a.png"},
            {"asset_id": "no_image", "name": "Has Name", "category": "tops", "gender_support": ["male"]},
        ]
    }
    p = tmp_path / "r.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    rows, report = meghna.normalize(p)
    assert len(rows) == 0
    assert report["summary"]["rejected"] == 2
    reasons = {item["reason"] for item in report["rejected"]}
    assert "name" in reasons
    assert "image_url_or_asset_path" in reasons


def test_meghna_walks_directory(tmp_path: Path):
    sub = tmp_path / "lib"
    sub.mkdir()
    (sub / "a.json").write_text(
        json.dumps({"category": "tops", "assets": [{"asset_id": "a", "name": "A", "subcategory": "shirt", "gender_support": ["male"], "image_url": "https://x/a.png"}]}),
        encoding="utf-8",
    )
    (sub / "b.json").write_text(
        json.dumps([{"asset_id": "b", "name": "B", "category": "bags", "gender": "female", "image_url": "https://x/b.png"}]),
        encoding="utf-8",
    )
    rows, report = meghna.normalize(sub)
    assert report["summary"]["total"] == 2
    assert report["summary"]["files_scanned"] == 2


# ---------- Importer ----------

def test_importer_normalize_keeps_allowed_fields_only():
    from scripts import import_style_assets as imp

    payload = imp._normalize(
        {
            "asset_id": "x",
            "name": "X",
            "category": "top",
            "image_url": "https://x/x.png",
            "tags": "a, b",
            "asset_type": "reference",  # not in ALLOWED_FIELDS
        }
    )
    assert "asset_type" not in payload
    assert payload["tags"] == ["a", "b"]
    assert payload["asset_id"] == "x"


def test_importer_skips_missing_image_url(tmp_path: Path):
    from scripts import import_style_assets as imp

    p = tmp_path / "x.json"
    p.write_text(
        json.dumps(
            [
                {"asset_id": "ok", "name": "Ok", "category": "top", "image_url": "https://x/ok.png"},
                {"asset_id": "skip", "name": "Skip", "category": "top"},
            ]
        ),
        encoding="utf-8",
    )
    stats = imp.import_assets(p, dry_run=True)
    assert stats["scanned"] == 2
    assert stats["failed"] == 1
