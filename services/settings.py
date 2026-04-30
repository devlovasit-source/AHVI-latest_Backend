import os
from typing import Self

from pydantic import BaseModel, Field, model_validator


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str) -> list[str]:
    raw = str(os.getenv(name, default))
    parts = [item.strip() for item in raw.split(",")]
    return [item for item in parts if item]


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return str(default or "").strip()


class AppSettings(BaseModel):
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_window_seconds: int = Field(default=60)
    rate_limit_max_requests: int = Field(default=120)
    rate_limit_require_redis: bool = Field(default=False)
    rate_limit_fail_closed: bool = Field(default=False)
    try_on_daily_limit: int = Field(default=2)
    upload_max_bytes: int = Field(default=20 * 1024 * 1024)
    auth_cache_ttl_seconds: int = Field(default=30)
    auth_required: bool = Field(default=True)
    redis_url: str = Field(default="redis://localhost:6379/0")
    cors_allowed_origins: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = Field(default=False)
    cors_allowed_methods: list[str] = Field(default_factory=lambda: ["*"])
    cors_allowed_headers: list[str] = Field(default_factory=lambda: ["*"])
    strict_router_loading: bool = Field(default=True)
    required_routers: list[str] = Field(default_factory=lambda: ["routers.chat", "routers.data"])

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.rate_limit_window_seconds <= 0:
            self.rate_limit_window_seconds = 60
        if self.rate_limit_max_requests <= 0:
            self.rate_limit_max_requests = 120
        if self.upload_max_bytes <= 0:
            self.upload_max_bytes = 20 * 1024 * 1024
        if self.auth_cache_ttl_seconds <= 0:
            self.auth_cache_ttl_seconds = 30
        if self.try_on_daily_limit <= 0:
            self.try_on_daily_limit = 2
        return self

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            rate_limit_enabled=_env_bool("RATE_LIMIT_ENABLED", True),
            rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
            rate_limit_max_requests=int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "120")),
            rate_limit_require_redis=_env_bool("RATE_LIMIT_REQUIRE_REDIS", False),
            rate_limit_fail_closed=_env_bool("RATE_LIMIT_FAIL_CLOSED", False),
            try_on_daily_limit=int(os.getenv("TRY_ON_DAILY_LIMIT", "2")),
            upload_max_bytes=int(os.getenv("UPLOAD_MAX_BYTES", str(20 * 1024 * 1024))),
            auth_cache_ttl_seconds=int(os.getenv("AUTH_CACHE_TTL_SECONDS", "30")),
            auth_required=_env_bool("AUTH_REQUIRED", True),
            redis_url=_env_first(
                "REDIS_URL",
                "REDIS_PRIVATE_URL",
                "REDIS_TLS_URL",
                "UPSTASH_REDIS_URL",
                "RAILWAY_REDIS_URL",
                default="redis://localhost:6379/0",
            ),
            cors_allowed_origins=_env_csv("CORS_ALLOWED_ORIGINS", "*"),
            cors_allow_credentials=_env_bool("CORS_ALLOW_CREDENTIALS", False),
            cors_allowed_methods=_env_csv("CORS_ALLOWED_METHODS", "*"),
            cors_allowed_headers=_env_csv("CORS_ALLOWED_HEADERS", "*"),
            strict_router_loading=_env_bool("STRICT_ROUTER_LOADING", True),
            required_routers=_env_csv("REQUIRED_ROUTERS", "routers.chat,routers.data"),
        )


settings = AppSettings.from_env()
