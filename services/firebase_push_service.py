from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional

try:
    import firebase_admin
    from firebase_admin import credentials, messaging
except Exception:  # pragma: no cover
    firebase_admin = None
    credentials = None
    messaging = None


def _load_service_account() -> dict | None:
    """
    Supported env vars:
    - FIREBASE_SERVICE_ACCOUNT_JSON: raw JSON string
    - FIREBASE_SERVICE_ACCOUNT_JSON_B64: base64 of JSON
    - FIREBASE_SERVICE_ACCOUNT_PATH: path to JSON file (Railway volume)
    """
    raw_json = str(os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")).strip()
    if raw_json:
        try:
            return json.loads(raw_json)
        except Exception:
            return None

    raw_b64 = str(os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON_B64", "")).strip()
    if raw_b64:
        try:
            decoded = base64.b64decode(raw_b64).decode("utf-8", errors="ignore")
            return json.loads(decoded)
        except Exception:
            return None

    path = str(os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "")).strip()
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    return None


class FirebasePushService:
    def __init__(self) -> None:
        self._ready = False
        self._init_error: str | None = None
        self._app = None

        if firebase_admin is None or credentials is None or messaging is None:
            self._init_error = "firebase_admin not installed"
            return

        sa = _load_service_account()
        if not sa:
            self._init_error = "missing firebase service account credentials"
            return

        try:
            cred = credentials.Certificate(sa)
            # Use a named app to avoid "already exists" issues in reloads.
            self._app = firebase_admin.initialize_app(cred, name="ahvi-fcm")
            self._ready = True
        except ValueError:
            # App already initialized (e.g. in hot reload); reuse default app.
            try:
                self._app = firebase_admin.get_app(name="ahvi-fcm")
                self._ready = True
            except Exception as exc:
                self._init_error = str(exc)
        except Exception as exc:
            self._init_error = str(exc)

    def ready(self) -> bool:
        return bool(self._ready)

    def status(self) -> dict:
        return {"ready": self.ready(), "error": self._init_error}

    def send_to_tokens(
        self,
        *,
        tokens: List[str],
        title: str,
        body: str,
        data: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if not self.ready():
            return {"success": False, "error": self._init_error or "not ready", "sent": 0, "failed": len(tokens)}

        safe_tokens = [str(t).strip() for t in (tokens or []) if str(t).strip()]
        if not safe_tokens:
            return {"success": True, "sent": 0, "failed": 0}

        # FCM data payload values must be strings.
        safe_data: Dict[str, str] = {}
        for k, v in (data or {}).items():
            if k is None:
                continue
            safe_data[str(k)] = "" if v is None else str(v)

        message = messaging.MulticastMessage(
            tokens=safe_tokens,
            notification=messaging.Notification(title=str(title or ""), body=str(body or "")),
            data=safe_data or None,
        )
        try:
            resp = messaging.send_multicast(message, app=self._app)
            return {
                "success": True,
                "sent": int(resp.success_count),
                "failed": int(resp.failure_count),
                "responses": [
                    {"success": bool(r.success), "exception": str(r.exception) if r.exception else None}
                    for r in (resp.responses or [])
                ],
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "sent": 0, "failed": len(safe_tokens)}


firebase_push_service = FirebasePushService()

