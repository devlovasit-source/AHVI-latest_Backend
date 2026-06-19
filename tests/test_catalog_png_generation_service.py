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


def _data_uri(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def test_clean_cutout_generates_transparent_catalog_png_without_provider(monkeypatch):
    monkeypatch.setenv("CATALOG_PROVIDER", "disabled")
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


def test_needs_review_forces_vertex_imagen(monkeypatch):
    calls = []

    class _Provider(pngsvc.CatalogProvider):
        name = "vertex_imagen"

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
    assert result["catalog_provider"] == "vertex_imagen"


def test_full_image_person_risk_forces_vertex_imagen(monkeypatch):
    calls = []

    class _Provider(pngsvc.CatalogProvider):
        name = "vertex_imagen"

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
    assert result["catalog_provider"] == "vertex_imagen"


def test_hanger_or_selfie_metadata_does_not_block_save_and_falls_back(monkeypatch):
    monkeypatch.setenv("CATALOG_PROVIDER", "disabled")
    raw = _garment_png(color=(180, 130, 95, 255))

    result = pngsvc.generate_catalog_png(
        raw,
        item_metadata={
            "item_id": "hanger-1",
            "name": "Mirror selfie hanger shirt",
            "category": "Tops",
            "source": "mirror_selfie_hanger",
        },
    )

    assert result["success"] is True
    assert result["status"] == "fallback_cutout"
    assert result["catalog_png_bytes"]
    assert result["catalog_quality_score"] < 82


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


def test_vertex_imagen_opaque_generated_output_is_accepted_for_demo(monkeypatch):
    class _Provider(pngsvc.CatalogProvider):
        name = "vertex_imagen"

        def generate(self, **kwargs):
            return pngsvc.CatalogProviderResult(
                True,
                image_bytes=_opaque_product_png(color=(190, 60, 60)),
                provider=self.name,
            )

    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())

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
    assert result["status"] == "catalog_generated"
    assert result["catalog_provider"] == "vertex_imagen"
    assert result["reason"] == "demo_accept_background"
    assert result["validation"]["checks"]["alpha_or_palette_mode"] is False


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
    raw = _garment_png(color=(180, 130, 95, 255))

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
    assert item["catalogQualityScore"] is not None
    assert item["category"] == "Ethnic Wear"
    assert item["sub_category"] == "Kurta"
    assert item["pattern"] == "woven"


def test_router_hook_passes_crop_risk_metadata_to_catalog_provider(monkeypatch):
    from routers import wardrobe_capture as wc

    provider_calls = []

    class _Provider(pngsvc.CatalogProvider):
        name = "vertex_imagen"

        def generate(self, **kwargs):
            provider_calls.append(kwargs)
            return pngsvc.CatalogProviderResult(
                True, image_bytes=kwargs["cutout_bytes"], provider=self.name
            )

    class _R2:
        def upload_catalog_png(self, *, file_id, image_bytes):
            return {
                "catalog_png_file_name": f"catalog_{file_id}.png",
                "catalog_png_url": f"https://cdn.test/catalog_{file_id}.png",
                "normalized_url": f"https://cdn.test/catalog_{file_id}.png",
            }

    monkeypatch.setenv("ENABLE_CATALOG_GENERATION", "true")
    monkeypatch.setenv("CATALOG_PROVIDER", "disabled")
    monkeypatch.setattr(pngsvc, "_provider_for", lambda name: _Provider())
    monkeypatch.setattr(wc, "R2Storage", lambda: _R2())

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

    assert provider_calls, "full_image_person_risk must force vertex provider"
    assert item["normalized_url"] == "https://cdn.test/catalog_shorts-risk-1.png"
    assert item["catalogStatus"] == "catalog_generated"


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
