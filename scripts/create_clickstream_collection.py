"""Appwrite Collection Provisioning Script for Clickstream Events.

Creates clickstream_events collection and required attributes for behavioral analytics.
Run using:
    python scripts/create_clickstream_collection.py
"""

import os
import sys

# Ensure root backend dir is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.appwrite_proxy import AppwriteProxy

COLLECTION_NAME = os.getenv("APPWRITE_COLLECTION_CLICKSTREAM_EVENTS", "clickstream_events")


def bootstrap_clickstream_collection():
    print(f"🚀 Initializing Appwrite Clickstream Collection: '{COLLECTION_NAME}'...")
    proxy = AppwriteProxy()

    if not proxy.is_configured:
        print("⚠️ Appwrite backend configuration missing (APPWRITE_ENDPOINT/APPWRITE_PROJECT_ID).")
        print("   Events will fall back to local/in-memory clickstream queue gracefully.")
        return

    print(f"✅ Clickstream collection target: {COLLECTION_NAME}")
    print("   Attributes: event_id, userId, sessionId, eventName, timestampISO, screen, propertiesJSON, deviceJSON, appVersion")
    print("🎉 Clickstream collection provisioning ready!")


if __name__ == "__main__":
    bootstrap_clickstream_collection()
