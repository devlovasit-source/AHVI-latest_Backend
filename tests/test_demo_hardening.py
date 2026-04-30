import asyncio
import importlib
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _request_for_user(user_id="user_auth"):
    return SimpleNamespace(state=SimpleNamespace(user={"user_id": user_id}))


class DemoHardeningTests(unittest.TestCase):
    def test_qdrant_enabled_false_when_unconfigured(self):
        with patch.dict(os.environ, {"QDRANT_URL": ""}, clear=False):
            from services.qdrant_service import QdrantService

            service = QdrantService()

        self.assertFalse(service.enabled())
        self.assertFalse(service.status()["enabled"])

    def test_duplicate_check_degrades_when_qdrant_unavailable(self):
        from routers import data

        with patch.object(data.qdrant_service, "enabled", return_value=False):
            result = data.check_outfit_duplicate(
                _request_for_user("user_auth"),
                data.OutfitDuplicateCheckRequest(data={}, user_id="user_auth"),
            )

        self.assertFalse(result["checked"])
        self.assertFalse(result["duplicate"]["is_duplicate"])

    def test_list_documents_uses_authenticated_user(self):
        from routers import data

        captured = {}

        def fake_list_documents(resource, **kwargs):
            captured.update(kwargs)
            return {"documents": [], "meta": {"mode": "query", "total": 0}}

        with patch.object(data.proxy, "list_documents", side_effect=fake_list_documents):
            data.list_documents(_request_for_user("user_auth"), "outfits")

        self.assertEqual(captured["user_id"], "user_auth")

    def test_create_document_stamps_authenticated_user(self):
        from routers import data

        captured = {}

        def fake_create(*, resource, payload, document_id):
            captured["resource"] = resource
            captured["payload"] = payload
            captured["document_id"] = document_id
            return {"$id": "doc_1", **payload}

        with patch.object(data.qdrant_service, "enabled", return_value=False), patch.object(
            data, "_create_document_with_schema_retries", side_effect=fake_create
        ):
            result = data.create_document(
                _request_for_user("user_auth"),
                data.CreateRequest(resource="plans", data={"title": "Demo"}),
            )

        self.assertEqual(result["document"]["userId"], "user_auth")
        self.assertEqual(captured["payload"]["userId"], "user_auth")

    def test_mismatched_user_ids_are_rejected(self):
        from routers import data

        with self.assertRaises(HTTPException) as exc:
            data.list_documents(_request_for_user("user_auth"), "outfits", user_id="other")
        self.assertEqual(exc.exception.status_code, 403)

        with self.assertRaises(HTTPException) as exc:
            data.create_document(
                _request_for_user("user_auth"),
                data.CreateRequest(resource="plans", data={"userId": "other"}),
            )
        self.assertEqual(exc.exception.status_code, 403)

        with self.assertRaises(HTTPException) as exc:
            data.get_document(_request_for_user("user_auth"), "users", "other")
        self.assertEqual(exc.exception.status_code, 403)

    def test_direct_document_routes_reject_other_owners(self):
        from routers import data

        with patch.object(
            data.proxy,
            "get_document",
            return_value={"$id": "doc_1", "userId": "other"},
        ):
            with self.assertRaises(HTTPException) as exc:
                data.get_document(_request_for_user("user_auth"), "outfits", "doc_1")
            self.assertEqual(exc.exception.status_code, 403)

            with self.assertRaises(HTTPException) as exc:
                data.update_document(
                    _request_for_user("user_auth"),
                    "doc_1",
                    data.UpdateRequest(resource="outfits", data={"name": "Updated"}),
                )
            self.assertEqual(exc.exception.status_code, 403)

            with self.assertRaises(HTTPException) as exc:
                data.delete_document(
                    _request_for_user("user_auth"),
                    data.DeleteRequest(resource="outfits", document_id="doc_1"),
                )
            self.assertEqual(exc.exception.status_code, 403)

    def test_rate_limit_identity_uses_user_id(self):
        import main

        captured = {}

        async def fake_is_ready():
            return False

        async def fake_check_rate_limit(*, bucket_key, max_requests, window_seconds):
            captured["bucket_key"] = bucket_key
            return True, 119

        async def fake_call_next(request):
            return Response("ok")

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/data/outfits",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "scheme": "http",
            "server": ("testserver", 80),
            "query_string": b"",
        }
        request = Request(scope)
        request.state.user = {"user_id": "user_auth"}

        with patch.object(main, "is_redis_rate_limit_ready", side_effect=fake_is_ready), patch.object(
            main, "check_rate_limit", side_effect=fake_check_rate_limit
        ):
            asyncio.run(main.rate_limit_middleware(request, fake_call_next))

        self.assertTrue(captured["bucket_key"].startswith("user_auth:"))

    def test_appwrite_proxy_ignores_frontend_prefixed_admin_key(self):
        env = {
            "APPWRITE_API_KEY": "",
            "APPWRITE_KEY": "",
            "EXPO_PUBLIC_APPWRITE_API_KEY": "public-key",
        }
        with patch.dict(os.environ, env, clear=False):
            import services.appwrite_proxy as appwrite_proxy

            importlib.reload(appwrite_proxy)
            proxy = appwrite_proxy.AppwriteProxy()

        self.assertEqual(proxy.api_key, "")


if __name__ == "__main__":
    unittest.main()
