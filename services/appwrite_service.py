import os
from functools import lru_cache
from typing import Optional

from appwrite.client import Client
from appwrite.services.account import Account
from appwrite.services.databases import Databases
from dotenv import load_dotenv


# =========================
# LOAD ENV (ONCE)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


# =========================
# ENV HELPERS
# =========================
def _env_first(*keys: str, default: Optional[str] = None) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


# =========================
# CONFIG (VALIDATED)
# =========================
APPWRITE_ENDPOINT = _env_first(
    "APPWRITE_ENDPOINT",
    "EXPO_PUBLIC_APPWRITE_ENDPOINT",
    default="https://cloud.appwrite.io/v1",
)

APPWRITE_PROJECT_ID = _env_first(
    "APPWRITE_PROJECT_ID",
    "APPWRITE_PROJECT",
    "EXPO_PUBLIC_APPWRITE_PROJECT_ID",
)

APPWRITE_API_KEY = _env_first("APPWRITE_API_KEY", "APPWRITE_KEY")


def is_appwrite_configured() -> bool:
    return bool(str(APPWRITE_PROJECT_ID or "").strip())


# =========================
# BASE CLIENT BUILDER
# =========================
def _create_base_client() -> Client:
    if not is_appwrite_configured():
        raise RuntimeError("APPWRITE_PROJECT_ID is not configured")
    client = Client()
    client.set_endpoint(APPWRITE_ENDPOINT)
    client.set_project(APPWRITE_PROJECT_ID)
    return client


# =========================
# 🔥 ADMIN CLIENT (CACHED)
# =========================
@lru_cache(maxsize=1)
def get_admin_client() -> Client:
    client = _create_base_client()

    if APPWRITE_API_KEY:
        client.set_key(APPWRITE_API_KEY)

    return client


# =========================
# 🔥 SERVICES (ADMIN)
# =========================
@lru_cache(maxsize=1)
def get_account_service() -> Account:
    return Account(get_admin_client())


@lru_cache(maxsize=1)
def get_database_service() -> Databases:
    return Databases(get_admin_client())


# =========================
# 🔥 JWT CLIENT (PER REQUEST)
# =========================
def build_account_for_jwt(token: str) -> Account:
    token = str(token or "").strip()

    if not token:
        raise ValueError("JWT token is empty")

    client = _create_base_client()
    client.set_jwt(token)

    return Account(client)
