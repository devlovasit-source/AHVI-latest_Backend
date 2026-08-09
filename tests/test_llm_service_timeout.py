from services import llm_service


def test_gemini_client_uses_call_specific_timeout_in_milliseconds(monkeypatch):
    captured = {}

    class FakeTypes:
        @staticmethod
        def HttpOptions(**kwargs):
            captured["http_options"] = kwargs
            return kwargs

    class FakeGenai:
        @staticmethod
        def Client(**kwargs):
            captured["client"] = kwargs
            return object()

    monkeypatch.setattr(llm_service, "AI_PROVIDER", "gemini")
    monkeypatch.setattr(llm_service, "types", FakeTypes)
    monkeypatch.setattr(llm_service, "genai", FakeGenai)
    llm_service._gemini_clients.clear()

    llm_service._get_gemini_client(timeout_seconds=12)

    assert captured["http_options"] == {"api_version": "v1", "timeout": 12000}
    assert captured["client"]["http_options"] == captured["http_options"]
