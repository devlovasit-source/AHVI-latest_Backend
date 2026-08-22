"""Unit & Contract Tests for Canonical Wardrobe Interactions & Lifecycle System."""

import unittest
from unittest.mock import MagicMock, patch

from services.wear_event_service import WearEventService, build_idempotency_key
from services.wardrobe_item_service import WardrobeItemService


class TestWardrobeInteractions(unittest.TestCase):
    def test_idempotency_key_builder(self):
        key = build_idempotency_key("user_123", "wardrobe_item", "item", "pink_jacket", "2026-08-17")
        self.assertEqual(key, "user_123:wardrobe_item:item:pink_jacket:2026-08-17")

    @patch("services.wear_event_service._proxy")
    def test_wear_event_service_duplicate_detection(self, mock_proxy_fn):
        mock_proxy = MagicMock()
        mock_proxy_fn.return_value = mock_proxy

        idem_key = "u1:wardrobe_item:item:item_99:2026-08-17"
        doc_id = idem_key.replace(":", "_")
        mock_proxy.get_document.return_value = {
            "$id": doc_id,
            "idempotencyKey": idem_key,
            "itemIds": ["item_99"],
            "occurredAtISO": "2026-08-17T08:30:00+05:30",
        }
        mock_proxy.list_documents.return_value = [
            {
                "$id": doc_id,
                "idempotencyKey": idem_key,
                "itemIds": ["item_99"],
                "occurredAtISO": "2026-08-17T08:30:00+05:30",
                "revokedAtISO": None,
            }
        ]

        service = WearEventService()
        res = service.record_wear(
            user_id="u1",
            item_ids=["item_99"],
            source="wardrobe_item",
            entity_type="item",
            entity_id="item_99",
            occurred_at="2026-08-17T08:30:00+05:30",
        )

        self.assertTrue(res["duplicate"])
        self.assertFalse(res["recorded"])
        self.assertEqual(res["summary"]["total_wears"], 1)

    @patch("services.wardrobe_item_service.AppwriteProxy")
    def test_favorite_state_diffing(self, mock_proxy_class):
        mock_proxy = MagicMock()
        mock_proxy_class.return_value = mock_proxy

        mock_proxy.get_document.return_value = {
            "$id": "item_123",
            "userId": "user_1",
            "name": "Pink Jacket",
            "is_favorite": False,
        }

        service = WardrobeItemService()
        service.proxy = mock_proxy

        res1 = service.set_favorite(user_id="user_1", item_id="item_123", is_favorite=True)
        self.assertTrue(res1["is_favorite"])
        mock_proxy.update_document.assert_called_once()

        mock_proxy.get_document.return_value["is_favorite"] = True
        mock_proxy.update_document.reset_mock()
        res2 = service.set_favorite(user_id="user_1", item_id="item_123", is_favorite=True)
        self.assertTrue(res2["is_favorite"])
        mock_proxy.update_document.assert_called_once()

    @patch("services.wardrobe_item_service.AppwriteProxy")
    @patch("services.wardrobe_item_service.QdrantService")
    @patch("services.wardrobe_item_service.R2Storage")
    def test_idempotent_soft_delete(self, mock_r2_class, mock_qdrant_class, mock_proxy_class):
        mock_proxy = MagicMock()
        mock_proxy_class.return_value = mock_proxy
        mock_qdrant = MagicMock()
        mock_qdrant_class.return_value = mock_qdrant
        mock_r2 = MagicMock()
        mock_r2_class.return_value = mock_r2

        mock_proxy.get_document.return_value = {
            "$id": "item_123",
            "userId": "user_1",
            "status": "active",
        }

        service = WardrobeItemService()
        service.proxy = mock_proxy

        res = service.delete_item(user_id="user_1", item_id="item_123")
        self.assertTrue(res)

        mock_proxy.get_document.return_value["status"] = "deleted"
        mock_proxy.update_document.reset_mock()
        res2 = service.delete_item(user_id="user_1", item_id="item_123")
        self.assertTrue(res2)
        mock_proxy.update_document.assert_not_called()


if __name__ == "__main__":
    unittest.main()
