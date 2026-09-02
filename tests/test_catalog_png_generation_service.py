import base64
import io
from types import SimpleNamespace

from PIL import Image, ImageDraw

from services import catalog_png_generation_service as pngsvc


def _garment_png(w=800, h=900, color=(40, 90, 200, 255), margin=0.18):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([w * margin, h * margin, w * (1 - margin), h * (1 - margin)], fill=color)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _opaque_product_png(w=900, h=900, color=(40, 90, 200), margin=0.18):
    im = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(im)
    d.rectangle([w * margin, h * margin, w * (1 - margin), h * (1 - margin)], fill=color)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _black_frame_png(w=900, h=900):
    im = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([w * 0.12, h * 0.12, w * 0.88, h * 0.88], fill=(250, 250, 250))
    d.rectangle([w * 0.35, h * 0.20, w * 0.65, h * 0.82], fill=(30, 90, 180))
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _blank_png(w=900, h=900):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _inset_black_frame_png(w=900, h=900):
    # White outer margin, black rectangular frame inset ~8-12%, white center
    # with a colored garment so the inner content survives cropping.
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([w * 0.08, h * 0.08, w * 0.92, h * 0.92], fill=(0, 0, 0))
    d.rectangle([w * 0.12, h * 0.12, w * 0.88, h * 0.88], fill=(255, 255, 255))
    d.rectangle([w * 0.38, h * 0.30, w * 0.62, h * 0.74], fill=(60, 140, 90))
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _vertical_side_bars_png(w=900, h=900):
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([w * 0.04, 0, w * 0.09, h], fill=(0, 0, 0))
    d.rectangle([w * 0.91, 0, w * 0.96, h], fill=(0, 0, 0))
    d.rectangle([w * 0.40, h * 0.25, w * 0.60, h * 0.78], fill=(60, 100, 170))
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _thin_black_edge_png(w=900, h=900):
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w - 1, h - 1], outline=(0, 0, 0), width=2)
    d.rectangle([w * 0.40, h * 0.25, w * 0.60, h * 0.78], fill=(60, 100, 170))
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _dark_garment_blob_png(w=900, h=900):
    # Dark garment near center, no continuous rectangular frame structure.
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([w * 0.30, h * 0.30, w * 0.70, h * 0.70], fill=(20, 20, 20))
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def test_inset_black_frame_detected():
    metrics = pngsvc._black_frame_metrics(_inset_black_frame_png())
    assert metrics["detected"] is True
    assert metrics["frame_type"] in {"inset_frame", "outer_border"}


def test_vertical_black_side_bars_detected():
    metrics = pngsvc._black_frame_metrics(_vertical_side_bars_png())
    assert metrics["detected"] is True
    assert metrics["frame_type"] in {"vertical_side_bars", "inset_frame", "outer_border"}


def test_thin_black_edge_detected():
    metrics = pngsvc._black_frame_metrics(_thin_black_edge_png())
    assert metrics["detected"] is True


def test_inset_black_frame_cropped():
    cropped_bytes, cropped = pngsvc._crop_black_frame(_inset_black_frame_png())
    assert cropped is True
    assert pngsvc._black_frame_metrics(cropped_bytes)["detected"] is False


def test_dark_garment_not_false_positive():
    metrics = pngsvc._black_frame_metrics(_dark_garment_blob_png())
    assert metrics["detected"] is False


def test_unresolved_black_frame_score_capped():
    validation = pngsvc.validate_catalog_png(
        _inset_black_frame_png(),
        original_bytes=_inset_black_frame_png(),
        item_metadata={"category": "Tops"},
    )
    assert validation["ok"] is False
    assert validation["reason"] == "black_frame_unresolved"
    assert validation["score"] <= 44
    assert validation["checks"]["no_black_frame"] is False


def test_demo_relaxation_does_not_accept_black_frame():
    validation = {
        "generated": True,
        "score": 78,
        "reason": "black_frame_unresolved",
        "checks": {
            "no_face": True,
            "no_human": True,
            "no_mannequin": True,
            "category_matches_original": True,
            "image_size_valid": True,
            "no_black_frame": False,
        },
    }
    assert pngsvc._vertex_demo_accepts_generated_validation(validation) is False


def _data_uri(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def test_dress_catalog_prompt_removes_hanger_and_reconstructs_neckline():
    prompt = pngsvc._build_catalog_prompt(
        "Dresses",
        {
            "name": "Red Polka Dot Dress",
            "sub_category": "sleeveless dress",
            "color_name": "red",
            "pattern": "white polka dot",
        },
    )

    assert "The dress is the product and must remain fully visible." in prompt
    assert "The dress must remain the dominant object" in prompt
    assert "- hanger" in prompt
    assert "- hook" in prompt
    assert "- clothing rod" in prompt
    assert "- shoulder seams" in prompt
    assert "- neckline" in prompt
    assert "- armholes" in prompt
    assert "- upper bodice" in prompt
    assert "- remove the garment" in prompt
    assert "category = dress" in prompt
    assert "subcategory = sleeveless dress" in prompt
    assert "color = red" in prompt
    assert "pattern = white polka dot" in prompt
    assert "The final image must still be a red sleeveless dress white polka dot." in prompt


def test_top_catalog_prompt_reconstructs_shoulders_and_sleeves():
    prompt = pngsvc._build_catalog_prompt("Tops", {"name": "White Shirt"})

    assert "upper-body garment" in prompt
    assert "collar" in prompt
    assert "shoulder" in prompt
    assert "sleeve" in prompt


def test_accessory_catalog_prompt_preserves_shape_without_lifestyle_imagery():
    prompt = pngsvc._build_catalog_prompt("Accessories", {"name": "Leather Belt"})

    assert "professional fashion e-commerce image editor" in prompt
    assert "Preserve exactly:" in prompt
    assert "exact garment preservation over creativity" in prompt
    assert "accessory" in prompt
    assert "Preserve the exact shape" in prompt
    assert "Do not add people" in prompt
    assert "lifestyle imagery" in prompt


def test_clean_cutout_generates_transparent_catalog_png_without_provider(monkeypatch):
    monkeypatch.setenv("CATALOG_PROVIDER", "disabled")
    monkeypatch.delenv("WARDROBE_CATALOG_PROVIDER", raising=False)
    raw = _garment_png()

    result = pngsvc.generate_catalog_png(
        raw,
        item_metadata={
            "item_id": "shirt-1",
            "name": "Blue Shirt",
            "category": "Tops",
            "sub_category": "Shirt",
            "pattern": "plain",
        },
    )

    assert result["success"] is True
    assert result["status"] == "catalog_ready"
    assert result["catalog_provider"] == "cutout"
    assert result["catalog_quality_score"] >= 80
    img = Image.open(io.BytesIO(result["catalog_png_bytes"]))
    assert img.mode == "RGBA"
    assert img.size == (1600, 1600)
    assert img.getpixel((0, 0))[3] == 0


def test_flat_lay_quality_gate_ok_still_invokes_nanobanana(monkeypatch):
    # Routing rule: a clean flat-lay (quality_gate_ok) MUST still go through
    # Nano Banana when the provider is nanobanana. Cutout is fallback only.
    calls = []

    class _Provider(pngsvc.CatalogProvider):
        name = "nanobanana"

        def generate(self, **kwargs):
            calls.append(kwargs)
            return pngsvc.CatalogProviderResult(
                True,
                image_bytes=kwargs["cutout_bytes"],
                provider=self.name,
            )

    monkeypatch.setenv("WARDROBE_CATALOG_PROVIDER", "nanobanana")
    monkeypatch.setenv("CATALOG_PROVIDER", "vertex_imagen")
    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())

    result = pngsvc.generate_catalog_png(
        _garment_png(),
        item_metadata={
            "item_id": "quality-ok-nano-1",
            "name": "Blue Shirt",
            "category": "Tops",
        },
    )

    assert calls, "nanobanana must run even when the cutout quality gate is ok"
    assert result["status"] == "catalog_generated"
    assert result["catalog_provider"] == "nanobanana"


def test_quality_gate_ok_still_runs_provider_for_nanobanana(monkeypatch):
    calls = []

    class _Provider(pngsvc.CatalogProvider):
        name = "nanobanana"

        def generate(self, **kwargs):
            calls.append(kwargs)
            return pngsvc.CatalogProviderResult(
                True, image_bytes=kwargs["cutout_bytes"], provider=self.name
            )

    monkeypatch.setenv("WARDROBE_CATALOG_PROVIDER", "nanobanana")
    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())

    result = pngsvc.generate_catalog_png(
        _garment_png(),
        item_metadata={
            "item_id": "quality-ok-log-1",
            "name": "Yellow T-Shirt",
            "category": "Tops",
        },
    )

    assert calls, "nanobanana must run for a clean flat-lay"
    assert result["status"] == "catalog_generated"


def test_fallback_cutout_true_still_generates_when_quality_ok(monkeypatch):
    # fallback_to_cutout only governs what happens AFTER generation fails; it
    # must not skip generation for a clean nanobanana flat-lay.
    calls = []

    class _Provider(pngsvc.CatalogProvider):
        name = "nanobanana"

        def generate(self, **kwargs):
            calls.append(kwargs)
            return pngsvc.CatalogProviderResult(True, image_bytes=kwargs["cutout_bytes"], provider=self.name)

    monkeypatch.setenv("WARDROBE_CATALOG_PROVIDER", "nanobanana")
    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())

    result = pngsvc.generate_catalog_png(
        _garment_png(),
        item_metadata={"item_id": "fallback-true-quality-ok", "name": "Yellow T-Shirt", "category": "Tops"},
        fallback_to_cutout=True,
    )

    assert calls, "generation must run even with fallback enabled on a clean flat-lay"
    assert result["status"] == "catalog_generated"
    assert result["catalog_provider"] == "nanobanana"


def test_clean_flatlay_generation_failure_no_fallback_does_not_save_cutout(monkeypatch):
    # Generation fails + fallback disabled -> must NOT silently save raw/cutout.
    class _Provider(pngsvc.CatalogProvider):
        name = "nanobanana"

        def generate(self, **kwargs):
            return pngsvc.CatalogProviderResult(False, reason="nano_down", provider=self.name)

    monkeypatch.setenv("WARDROBE_CATALOG_PROVIDER", "nanobanana")
    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())

    result = pngsvc.generate_catalog_png(
        _garment_png(),
        item_metadata={"item_id": "nofallback-1", "name": "Blue Shirt", "category": "Tops"},
        fallback_to_cutout=False,
    )

    assert result["success"] is False
    assert result["status"] == "catalog_failed"
    assert "catalog_png_bytes" not in result


def test_nanobanana_black_frame_output_is_cropped_before_acceptance(monkeypatch):
    class _Provider(pngsvc.CatalogProvider):
        name = "nanobanana"

        def generate(self, **kwargs):
            return pngsvc.CatalogProviderResult(
                True,
                image_bytes=_black_frame_png(),
                provider=self.name,
            )

    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())

    def _transparent_after_crop(image_bytes, *_args):
        assert pngsvc._black_frame_metrics(image_bytes)["detected"] is False
        return _garment_png(), "ok"

    monkeypatch.setattr(
        pngsvc, "_provider_output_to_transparent", _transparent_after_crop
    )

    result = pngsvc.generate_catalog_png(
        _garment_png(),
        provider="nanobanana",
        item_metadata={
            "item_id": "black-frame-1",
            "name": "Blue Shirt",
            "category": "Tops",
            "needs_review": True,
        },
    )

    assert result["success"] is True
    assert result["status"] == "catalog_generated"
    assert result["catalog_provider"] == "nanobanana"
    assert pngsvc._black_frame_metrics(result["catalog_png_bytes"])["detected"] is False


def test_nanobanana_blank_jewelry_output_blocks_cutout_fallback(monkeypatch):
    class _Provider(pngsvc.CatalogProvider):
        name = "nanobanana"

        def generate(self, **kwargs):
            return pngsvc.CatalogProviderResult(
                True,
                image_bytes=_blank_png(),
                provider=self.name,
            )

    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())

    result = pngsvc.generate_catalog_png(
        _garment_png(color=(230, 210, 120, 255)),
        provider="nanobanana",
        item_metadata={
            "item_id": "blank-jewelry-1",
            "name": "Gold Necklace",
            "category": "Jewelry",
            "needs_review": True,
        },
    )

    assert result["success"] is False
    assert result["status"] == "blocked_blank_catalog"
    assert result["catalog_provider"] == "nanobanana"
    assert result["reason"] in {"blank_transparent_catalog", "blank_flat_catalog", "tiny_accessory_catalog"}
    assert "catalog_png_bytes" not in result


def test_cutout_provider_can_still_use_quality_gate(monkeypatch):
    monkeypatch.setenv("WARDROBE_CATALOG_PROVIDER", "cutout")

    result = pngsvc.generate_catalog_png(
        _garment_png(),
        item_metadata={"item_id": "cutout-1", "name": "Blue Shirt", "category": "Tops"},
    )

    assert result["status"] == "catalog_ready"
    assert result["catalog_provider"] == "cutout"


def test_env_provider_nanobanana_routes_to_nano_provider(monkeypatch):
    monkeypatch.setenv("NANO_BANANA_CATALOG_MODEL", "nano-test")
    provider = pngsvc._provider_for("nanobanana")

    assert provider.name == "nanobanana"
    assert provider.model == "nano-test"


def test_needs_review_forces_nanobanana(monkeypatch):
    calls = []

    class _Provider(pngsvc.CatalogProvider):
        name = "nanobanana"

        def generate(self, **kwargs):
            calls.append(kwargs)
            return pngsvc.CatalogProviderResult(True, image_bytes=kwargs["cutout_bytes"], provider=self.name)

    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())
    raw = _garment_png()

    result = pngsvc.generate_catalog_png(
        raw,
        provider="disabled",
        item_metadata={
            "item_id": "review-1",
            "name": "Black Shorts",
            "category": "Bottoms",
            "sub_category": "Shorts",
            "needs_review": True,
        },
    )

    assert calls
    assert result["status"] == "catalog_generated"
    assert result["catalog_provider"] == "nanobanana"


def test_full_image_person_risk_forces_nanobanana(monkeypatch):
    calls = []

    class _Provider(pngsvc.CatalogProvider):
        name = "nanobanana"

        def generate(self, **kwargs):
            calls.append(kwargs)
            return pngsvc.CatalogProviderResult(True, image_bytes=kwargs["cutout_bytes"], provider=self.name)

    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())

    result = pngsvc.generate_catalog_png(
        _garment_png(),
        provider="disabled",
        item_metadata={
            "item_id": "person-risk-1",
            "name": "Black Shorts",
            "category": "Bottoms",
            "crop_quality": "full_image_person_risk",
        },
    )

    assert calls
    assert result["status"] == "catalog_generated"
    assert result["catalog_provider"] == "nanobanana"
    assert "waistband" in calls[0]["prompt"]
    assert "no hanger" in calls[0]["prompt"]
    assert "exact garment preservation over creativity" in calls[0]["prompt"]


def test_hanger_or_selfie_metadata_blocks_unsafe_cutout_fallback(monkeypatch):
    monkeypatch.setenv("CATALOG_PROVIDER", "disabled")
    raw = _garment_png(color=(30, 90, 180, 255))

    result = pngsvc.generate_catalog_png(
        raw,
        item_metadata={
            "item_id": "hanger-1",
            "name": "Mirror selfie hanger shirt",
            "category": "Tops",
            "source": "mirror_selfie_hanger",
        },
    )

    assert result["success"] is False
    assert result["status"] == "blocked_unsafe_fallback"
    assert result["catalog_provider"] == "nanobanana"
    assert result["reason"] == "unsafe_source_nanobanana_failed"
    assert "catalog_png_bytes" not in result


def test_provider_failure_returns_fallback_cutout(monkeypatch):
    monkeypatch.setenv("CATALOG_PROVIDER", "flux_kontext")
    monkeypatch.delenv("FLUX_KONTEXT_CATALOG_URL", raising=False)

    result = pngsvc.generate_catalog_png(
        _garment_png(),
        item_metadata={"item_id": "dress-1", "name": "Dress", "category": "Dresses"},
    )

    assert result["success"] is True
    assert result["status"] in {"catalog_ready", "fallback_cutout"}
    assert result["catalog_png_bytes"]


def test_vertex_imagen_provider_key_routes_to_vertex_provider(monkeypatch):
    monkeypatch.setenv("CATALOG_IMAGEN_MODEL", "imagen-test")
    provider = pngsvc._provider_for("vertex_imagen")
    assert provider.name == "vertex_imagen"
    assert provider.model == "imagen-test"


def test_vertex_imagen_missing_sdk_fails_open(monkeypatch):
    monkeypatch.setattr(pngsvc, "genai", None)
    monkeypatch.setattr(pngsvc, "types", None)
    provider = pngsvc.CatalogProviderVertexImagen()

    result = provider.generate(
        cutout_bytes=_garment_png(),
        prompt=pngsvc.CATALOG_PROMPT,
        item_metadata={"category": "Tops"},
        timeout=1,
    )

    assert result.success is False
    assert result.provider == "vertex_imagen"
    assert result.reason == "google_genai_unavailable"


def test_nanobanana_missing_sdk_fails_open(monkeypatch):
    monkeypatch.setattr(pngsvc, "genai", None)
    monkeypatch.setattr(pngsvc, "types", None)
    provider = pngsvc.CatalogProviderNanoBanana()

    result = provider.generate(
        cutout_bytes=_garment_png(),
        prompt=pngsvc.CATALOG_PROMPT,
        item_metadata={"category": "Tops"},
        timeout=1,
    )

    assert result.success is False
    assert result.provider == "nanobanana"
    assert result.reason == "google_genai_unavailable"


def test_nanobanana_generate_content_returns_catalog_png(monkeypatch):
    captured = {}
    generated = _opaque_product_png(color=(80, 120, 180))

    class _FakePart:
        @staticmethod
        def from_bytes(data, mime_type):
            return {"data": data, "mime_type": mime_type}

    class _FakeConfig(dict):
        def __init__(self, **kwargs):
            super().__init__(kwargs)

    class _FakeTypes:
        Part = _FakePart
        GenerateContentConfig = _FakeConfig
        HttpOptions = lambda *a, **kw: None

    class _Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    inline_data=SimpleNamespace(data=generated)
                                )
                            ]
                        )
                    )
                ]
            )

    provider = pngsvc.CatalogProviderNanoBanana()
    monkeypatch.setattr(pngsvc, "types", _FakeTypes)
    monkeypatch.setattr(provider, "_client", lambda: SimpleNamespace(models=_Models()))

    result = provider.generate(
        cutout_bytes=_garment_png(),
        prompt=pngsvc.CATALOG_PROMPT,
        item_metadata={"category": "Tops"},
        timeout=1,
    )

    assert result.success is True
    assert result.provider == "nanobanana"
    assert result.image_bytes.startswith(b"\x89PNG")
    assert captured["model"] == provider.model
    assert captured["contents"][0] == pngsvc.CATALOG_PROMPT
    assert captured["contents"][1]["mime_type"] == "image/png"


def test_vertex_imagen_edit_image_receives_valid_minimal_config(monkeypatch):
    captured = {}
    generated = _garment_png(color=(80, 120, 180, 255))

    class _Models:
        def edit_image(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                generated_images=[
                    SimpleNamespace(image=SimpleNamespace(image_bytes=generated))
                ]
            )

    provider = pngsvc.CatalogProviderVertexImagen()
    monkeypatch.setattr(provider, "_client", lambda: SimpleNamespace(models=_Models()))

    result = provider.generate(
        cutout_bytes=_garment_png(),
        prompt=pngsvc.CATALOG_PROMPT,
        item_metadata={"category": "Tops"},
        timeout=1,
    )

    assert result.success is True
    assert result.provider == "vertex_imagen"
    assert result.image_bytes.startswith(b"\x89PNG")
    config = captured["config"]
    assert config.number_of_images == 1
    assert config.output_mime_type == "image/png"
    assert config.add_watermark is False
    assert config.labels is None
    assert config.include_rai_reason is None


def test_vertex_imagen_config_validation_error_falls_back_to_cutout(monkeypatch):
    class _Provider(pngsvc.CatalogProviderVertexImagen):
        def _client(self):
            return SimpleNamespace(models=SimpleNamespace())

        def _edit_config(self):
            raise ValueError("2 validation errors for EditImageConfig\nbad field")

    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())

    result = pngsvc.generate_catalog_png(
        _garment_png(color=(180, 130, 95, 255)),
        provider="vertex_imagen",
        item_metadata={
            "item_id": "validation-1",
            "name": "Mirror selfie hanger shirt",
            "category": "Tops",
            "source": "mirror_selfie_hanger",
        },
    )

    assert result["success"] is True
    assert result["status"] == "fallback_cutout"
    assert result["catalog_provider"] == "vertex_imagen"
    assert "EditImageConfig" in result["reason"]
    assert result["catalog_png_bytes"]


def test_opaque_provider_output_is_discarded_when_rmbg_fails(monkeypatch):
    opaque_provider_bytes = _opaque_product_png(color=(190, 60, 60))

    class _Provider(pngsvc.CatalogProvider):
        name = "vertex_imagen"

        def generate(self, **kwargs):
            return pngsvc.CatalogProviderResult(
                True,
                image_bytes=opaque_provider_bytes,
                provider=self.name,
            )

    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())
    monkeypatch.setattr(
        pngsvc,
        "_provider_output_to_transparent",
        lambda *_args: (b"", "still_opaque"),
    )

    result = pngsvc.generate_catalog_png(
        _garment_png(color=(190, 60, 60, 255)),
        provider="vertex_imagen",
        item_metadata={
            "item_id": "opaque-1",
            "name": "Red Shirt",
            "category": "Tops",
            "needs_review": True,
        },
    )

    assert result["success"] is True
    assert result["status"] == "fallback_cutout"
    assert result["catalog_provider"] == "vertex_imagen"
    assert result["reason"] == "still_opaque"
    assert result["catalog_png_bytes"] != opaque_provider_bytes
    assert pngsvc.is_effectively_transparent(result["catalog_png_bytes"]) is True


def test_vertex_imagen_invalid_size_still_falls_back(monkeypatch):
    class _Provider(pngsvc.CatalogProvider):
        name = "vertex_imagen"

        def generate(self, **kwargs):
            return pngsvc.CatalogProviderResult(
                True,
                image_bytes=_opaque_product_png(w=300, h=300, color=(190, 60, 60)),
                provider=self.name,
            )

    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())

    result = pngsvc.generate_catalog_png(
        _garment_png(color=(190, 60, 60, 255)),
        provider="vertex_imagen",
        item_metadata={
            "item_id": "tiny-1",
            "name": "Red Shirt",
            "category": "Tops",
            "needs_review": True,
        },
    )

    assert result["success"] is True
    assert result["status"] == "fallback_cutout"
    assert result["catalog_provider"] == "vertex_imagen"
    assert result["catalog_png_bytes"]


def test_vertex_imagen_failed_call_falls_back_to_cutout(monkeypatch):
    class _Provider(pngsvc.CatalogProvider):
        name = "vertex_imagen"

        def generate(self, **kwargs):
            return pngsvc.CatalogProviderResult(False, reason="adc_missing", provider=self.name)

    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())
    raw = _garment_png(color=(30, 90, 180, 255))

    result = pngsvc.generate_catalog_png(
        raw,
        provider="vertex_imagen",
        item_metadata={
            "item_id": "hanger-2",
            "name": "Mirror selfie hanger shirt",
            "category": "Tops",
            "source": "mirror_selfie_hanger",
        },
    )

    assert result["success"] is True
    assert result["status"] == "fallback_cutout"
    assert result["reason"] == "adc_missing"
    assert result["catalog_png_bytes"]


def test_nanobanana_failure_returns_fallback_cutout(monkeypatch):
    class _Provider(pngsvc.CatalogProvider):
        name = "nanobanana"

        def generate(self, **kwargs):
            return pngsvc.CatalogProviderResult(False, reason="nano_down", provider=self.name)

    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())
    raw = _garment_png(color=(30, 90, 180, 255))

    result = pngsvc.generate_catalog_png(
        raw,
        provider="nanobanana",
        item_metadata={
            "item_id": "clean-3",
            "name": "Clean Blue Shirt",
            "category": "Tops",
            "needs_review": True,
        },
    )

    assert result["success"] is True
    assert result["status"] == "fallback_cutout"
    assert result["catalog_provider"] == "cutout"
    assert result["reason"] == "nano_down"
    assert result["catalog_png_bytes"]


def test_unsafe_source_nanobanana_failure_blocks_cutout_fallback(monkeypatch):
    class _Provider(pngsvc.CatalogProvider):
        name = "nanobanana"

        def generate(self, **kwargs):
            return pngsvc.CatalogProviderResult(False, reason="nanobanana_returned_no_image", provider=self.name)

    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())

    result = pngsvc.generate_catalog_png(
        _garment_png(color=(180, 130, 95, 255)),
        provider="nanobanana",
        item_metadata={
            "item_id": "unsafe-saree-1",
            "name": "Green Saree",
            "category": "Ethnic Wear",
            "source": "person_body_crop",
        },
    )

    assert result["success"] is False
    assert result["status"] == "blocked_unsafe_fallback"
    assert result["catalog_provider"] == "nanobanana"
    assert result["reason"] == "unsafe_source_nanobanana_failed"
    assert "catalog_png_bytes" not in result
    assert result["fallback_used"] is False


def test_unsafe_source_nanobanana_success_returns_generated(monkeypatch):
    class _Provider(pngsvc.CatalogProvider):
        name = "nanobanana"

        def generate(self, **kwargs):
            return pngsvc.CatalogProviderResult(True, image_bytes=kwargs["cutout_bytes"], provider=self.name)

    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())

    result = pngsvc.generate_catalog_png(
        _garment_png(color=(30, 120, 80, 255)),
        provider="nanobanana",
        item_metadata={
            "item_id": "unsafe-success-1",
            "name": "Green Saree",
            "category": "Ethnic Wear",
            "source": "person_body_crop",
        },
    )

    assert result["success"] is True
    assert result["status"] == "catalog_generated"
    assert result["catalog_provider"] == "nanobanana"


def test_http_imagen_and_flux_providers_still_use_http_envs():
    flux = pngsvc._provider_for("flux_kontext")
    imagen = pngsvc._provider_for("imagen")

    assert isinstance(flux, pngsvc.HttpCatalogProvider)
    assert flux.name == "flux_kontext"
    assert isinstance(imagen, pngsvc.HttpCatalogProvider)
    assert imagen.name == "imagen"


def test_validation_rejects_empty_catalog_png():
    empty = io.BytesIO()
    Image.new("RGBA", (600, 600), (0, 0, 0, 0)).save(empty, "PNG")

    result = pngsvc.validate_catalog_png(
        empty.getvalue(),
        original_bytes=empty.getvalue(),
        item_metadata={"category": "Tops"},
    )

    assert result["ok"] is False
    assert result["score"] == 0


def test_router_hook_uploads_catalog_png_and_preserves_metadata(monkeypatch):
    from routers import wardrobe_capture as wc

    class _R2:
        def upload_catalog_png(self, *, file_id, image_bytes):
            return {
                "catalog_png_file_name": f"catalog_{file_id}.png",
                "catalog_png_url": f"https://cdn.test/catalog_{file_id}.png",
                "normalized_url": f"https://cdn.test/catalog_{file_id}.png",
            }

    monkeypatch.setenv("ENABLE_CATALOG_GENERATION", "true")
    monkeypatch.setattr(wc, "R2Storage", lambda: _R2())

    item = {
        "item_id": "kurta-1",
        "name": "Blue Kurta",
        "category": "Ethnic Wear",
        "sub_category": "Kurta",
        "pattern": "woven",
        "color_name": "Blue",
        "masked_image_base64": _data_uri(_garment_png(color=(20, 80, 160, 255))),
    }

    wc._maybe_generate_catalog_image(item)

    assert item["normalized_url"] == "https://cdn.test/catalog_kurta-1.png"
    assert item["catalogStatus"] in {"catalog_ready", "fallback_cutout"}
    assert item.get("catalogProvider") in {"cutout", "nanobanana", "disabled"}
    assert item["catalogQualityScore"] is not None
    assert item["category"] == "Ethnic Wear"
    assert item["sub_category"] == "Kurta"
    assert item["pattern"] == "woven"


def test_router_hook_approved_item_uses_nanobanana_and_normalized_url(monkeypatch):
    from routers import wardrobe_capture as wc

    provider_calls = []

    class _Provider(pngsvc.CatalogProvider):
        name = "nanobanana"

        def generate(self, **kwargs):
            provider_calls.append(kwargs)
            return pngsvc.CatalogProviderResult(
                True,
                image_bytes=kwargs["cutout_bytes"],
                provider=self.name,
            )

    class _R2:
        def upload_catalog_png(self, *, file_id, image_bytes):
            return {
                "catalog_png_file_name": f"catalog_{file_id}.png",
                "catalog_png_url": f"https://cdn.test/catalog_{file_id}.png",
                "normalized_url": f"https://cdn.test/catalog_{file_id}.png",
            }

    monkeypatch.setenv("ENABLE_CATALOG_GENERATION", "true")
    monkeypatch.setenv("WARDROBE_CATALOG_PROVIDER", "nanobanana")
    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())
    monkeypatch.setattr(wc, "R2Storage", lambda: _R2())

    item = {
        "item_id": "shirt-approved-1",
        "name": "Blue Shirt",
        "category": "Tops",
        "sub_category": "Shirt",
        "validation_status": "ok",
        "source": "mirror_selfie_hanger",
        "masked_url": "https://cdn.test/masked_shirt-approved-1.png",
        "masked_image_base64": _data_uri(_garment_png(color=(30, 90, 180, 255))),
    }

    wc._maybe_generate_catalog_image(item)

    assert provider_calls
    assert item["normalized_url"] == "https://cdn.test/catalog_shirt-approved-1.png"
    assert item["normalizedUrl"] == "https://cdn.test/catalog_shirt-approved-1.png"
    assert item["masked_url"] == "https://cdn.test/masked_shirt-approved-1.png"
    assert item["catalogStatus"] == "catalog_generated"
    assert item["catalogProvider"] == "nanobanana"


def test_router_hook_full_image_fallback_skips_provider_and_needs_review(monkeypatch):
    # Full-image fallback (no clean single-garment crop) must NOT be sent to
    # Nano Banana — stylizing a full-frame/body shot yields a board-like image.
    # Skip generation and mark needs_review so the user re-shoots.
    from routers import wardrobe_capture as wc

    provider_calls = []

    class _Provider(pngsvc.CatalogProvider):
        name = "nanobanana"

        def generate(self, **kwargs):
            provider_calls.append(kwargs)
            return pngsvc.CatalogProviderResult(
                True, image_bytes=kwargs["cutout_bytes"], provider=self.name
            )

    monkeypatch.setenv("ENABLE_CATALOG_GENERATION", "true")
    monkeypatch.setenv("CATALOG_PROVIDER", "disabled")
    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())

    item = {
        "item_id": "shorts-risk-1",
        "name": "Black Shorts",
        "category": "Bottoms",
        "sub_category": "Shorts",
        "crop_source": "full_image_fallback",
        "crop_quality": "full_image_person_risk",
        "needs_review": True,
        "review_reason": "Needs cleaner photo",
        "requires_manual_entry": True,
        "source": "gemini_single_garment",
        "label_source": "gemini",
        "masked_image_base64": _data_uri(_garment_png(color=(20, 20, 20, 255))),
    }

    wc._maybe_generate_catalog_image(item)

    assert provider_calls == [], "full_image_fallback must NOT call the provider"
    assert item["catalogStatus"] == "catalog_skipped_full_frame"
    assert item["needs_review"] is True
    assert "normalized_url" not in item


def test_persistence_payload_maps_catalog_png_to_normalized_url(monkeypatch):
    from services.wardrobe_persistence_service import _build_appwrite_doc

    monkeypatch.setenv("ENABLE_CATALOG_GENERATION", "true")
    item = {
        "name": "Blue Shirt",
        "category": "Tops",
        "sub_category": "Shirt",
        "color_code": "#1450A0",
        "pattern": "plain",
        "occasions": ["Work"],
    }

    doc = _build_appwrite_doc(
        user_id="user-1",
        file_id="item-1",
        item=item,
        raw_url="https://cdn.test/raw_item-1.png",
        masked_url="https://cdn.test/cutout_item-1.png",
        normalized_url="https://cdn.test/normalized_item-1.png",
    )

    assert doc["image_url"] == "https://cdn.test/raw_item-1.png"
    assert doc["masked_url"] == "https://cdn.test/cutout_item-1.png"
    assert doc["normalized_url"] == "https://cdn.test/normalized_item-1.png"
    assert "original_image_url" not in doc
    assert "cutout_image_url" not in doc
    assert "catalog_png_url" not in doc
    assert "catalog_provider" not in doc
    assert "catalog_quality_score" not in doc
    assert "catalog_generation_version" not in doc


# ── Identity-drift gate ───────────────────────────────────────────────────
# _catalog_identity_drift() in routers/wardrobe_capture.py already hard-blocks
# on reason in {"identity_drift", "wrong_garment_type"}, but nothing ever
# produced that reason - the gate was unreachable. These tests cover the
# producer: _classify_identity_match() + its wiring into generate_catalog_png.


class _NanobananaEcho(pngsvc.CatalogProvider):
    """Fake nanobanana provider that just echoes the input cutout back as the
    'generated' image, so validate_catalog_png always scores it as ok - the
    only thing under test is the identity-check override, not the scorer."""

    name = "nanobanana"

    def generate(self, **kwargs):
        return pngsvc.CatalogProviderResult(
            True, image_bytes=kwargs["cutout_bytes"], provider=self.name
        )


def test_identity_check_disabled_by_default(monkeypatch):
    # Flag defaults off, matching every other vision gate in this module
    # (orientation, etc) - safe rollout, opt-in per environment.
    assert pngsvc._identity_check_enabled() is False
    assert pngsvc._identity_confidence_min() == 0.75


def test_classify_identity_match_fails_safe_without_client(monkeypatch):
    # No vertex creds / no genai module in test env -> must return "uncertain"
    # with confidence 0, NEVER a false "match" or false "mismatch".
    result = pngsvc._classify_identity_match(
        _garment_png(), _garment_png(), "footwear", {"name": "Sneaker"}
    )
    assert result["match"] == "uncertain"
    assert result["confidence"] == 0.0


def test_confident_identity_mismatch_forces_cutout_fallback(monkeypatch):
    monkeypatch.setenv("WARDROBE_CATALOG_PROVIDER", "nanobanana")
    monkeypatch.setenv("CATALOG_PROVIDER", "vertex_imagen")
    monkeypatch.setenv("CATALOG_IDENTITY_CHECK", "true")
    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _NanobananaEcho())
    monkeypatch.setattr(
        pngsvc,
        "_classify_identity_match",
        lambda *a, **k: {"match": "mismatch", "confidence": 0.9, "evidence": "wrong_shoe_type"},
    )

    result = pngsvc.generate_catalog_png(
        _garment_png(),
        item_metadata={"item_id": "shoe-1", "name": "White Sneaker", "category": "footwear"},
    )

    # Must NOT save the AI-regenerated image under a "catalog_generated"
    # status - falls back to the deterministic cutout instead.
    assert result["status"] == "fallback_cutout"
    assert result["reason"] == "identity_mismatch"
    assert result["catalog_provider"] != "nanobanana"


def test_confident_identity_match_allows_catalog_generated(monkeypatch):
    monkeypatch.setenv("WARDROBE_CATALOG_PROVIDER", "nanobanana")
    monkeypatch.setenv("CATALOG_PROVIDER", "vertex_imagen")
    monkeypatch.setenv("CATALOG_IDENTITY_CHECK", "true")
    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _NanobananaEcho())
    monkeypatch.setattr(
        pngsvc,
        "_classify_identity_match",
        lambda *a, **k: {"match": "match", "confidence": 0.95, "evidence": "ok"},
    )

    result = pngsvc.generate_catalog_png(
        _garment_png(),
        item_metadata={"item_id": "shoe-2", "name": "White Sneaker", "category": "footwear"},
    )

    assert result["status"] == "catalog_generated"
    assert result["catalog_provider"] == "nanobanana"


def test_identity_check_skipped_when_flag_off(monkeypatch):
    # Flag OFF must reproduce today's behavior exactly, even if the (mocked)
    # classifier would have reported a mismatch - opt-in rollout, zero
    # behavior change until an environment explicitly turns this on.
    monkeypatch.setenv("WARDROBE_CATALOG_PROVIDER", "nanobanana")
    monkeypatch.setenv("CATALOG_PROVIDER", "vertex_imagen")
    monkeypatch.setenv("CATALOG_IDENTITY_CHECK", "false")
    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _NanobananaEcho())
    calls = []

    def _spy(*a, **k):
        calls.append(1)
        return {"match": "mismatch", "confidence": 0.99, "evidence": "should_not_run"}

    monkeypatch.setattr(pngsvc, "_classify_identity_match", _spy)

    result = pngsvc.generate_catalog_png(
        _garment_png(),
        item_metadata={"item_id": "shoe-3", "name": "White Sneaker", "category": "footwear"},
    )

    assert not calls, "identity classifier must not run when the flag is off"
    assert result["status"] == "catalog_generated"


def test_low_confidence_identity_mismatch_does_not_block(monkeypatch):
    # A low-confidence mismatch must not reject - only a confident one
    # (>= CATALOG_IDENTITY_CONFIDENCE_MIN) should override "ok".
    monkeypatch.setenv("WARDROBE_CATALOG_PROVIDER", "nanobanana")
    monkeypatch.setenv("CATALOG_PROVIDER", "vertex_imagen")
    monkeypatch.setenv("CATALOG_IDENTITY_CHECK", "true")
    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _NanobananaEcho())
    monkeypatch.setattr(
        pngsvc,
        "_classify_identity_match",
        lambda *a, **k: {"match": "mismatch", "confidence": 0.4, "evidence": "unsure"},
    )

    result = pngsvc.generate_catalog_png(
        _garment_png(),
        item_metadata={"item_id": "shoe-4", "name": "White Sneaker", "category": "footwear"},
    )

    assert result["status"] == "catalog_generated"