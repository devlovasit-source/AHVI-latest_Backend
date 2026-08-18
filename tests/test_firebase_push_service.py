import os
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import services.firebase_push_service as firebase_module


class FirebasePushServiceTests(unittest.TestCase):
    @patch.object(firebase_module, "messaging")
    def test_send_to_tokens_deduplicates_tokens(self, mock_messaging):
        service = firebase_module.FirebasePushService.__new__(
            firebase_module.FirebasePushService
        )
        service._ready = True
        service._init_error = None
        service._app = MagicMock()

        mock_response = MagicMock()
        mock_response.success_count = 3
        mock_response.failure_count = 0
        mock_response.responses = []

        mock_messaging.send_each_for_multicast.return_value = mock_response

        service.send_to_tokens(
            tokens=[
                "token-a",
                "token-b",
                "token-a",
                "  token-b  ",
                "",
                "token-c",
            ],
            title="Test",
            body="Test notification",
        )

        message = mock_messaging.MulticastMessage.call_args.kwargs

        self.assertEqual(
            message["tokens"],
            ["token-a", "token-b", "token-c"],
        )

        mock_messaging.send_each_for_multicast.assert_called_once()