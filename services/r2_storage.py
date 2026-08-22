import logging
import os
import threading
from io import BytesIO
from typing import Dict, Optional
import uuid

try:
    from minio import Minio
except Exception:
    Minio = None

try:
    import urllib3
except Exception:
    urllib3 = None

try:
    from services.image_normalizer import normalize_wardrobe_image_bytes
except Exception:
    normalize_wardrobe_image_bytes = None


logger = logging.getLogger(__name__)


class R2StorageError(Exception):
    pass


def _env(name: str, fallback: str = "") -> str:
    return os.getenv(name, fallback)


def _load_local_env() -> None:
    """Walk parent dirs to populate env from .env files. Off by default.

    Set R2_LOAD_LOCAL_ENV=true to enable for local dev. On Cloud Run, env vars
    come from the platform; loading parent .env files there is unnecessary and
    can pollute os.environ mid-runtime.
    """
    if str(os.getenv("R2_LOAD_LOCAL_ENV", "false")).strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    cwd = os.getcwd()
    parent = os.path.dirname(cwd)
    candidate_paths = [
        os.path.join(cwd, ".env"),
        os.path.join(parent, ".env"),
        os.path.join(parent, "backend", ".env"),
        os.path.join(parent, "backend-master", ".env"),
        os.path.join(parent, "ahvi", ".env"),
    ]

    for env_path in candidate_paths:
        if not os.path.exists(env_path):
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and (
                        key not in os.environ
                        or not str(os.environ.get(key, "")).strip()
                    ):
                        os.environ[key] = value
        except Exception:
            logger.exception("r2 local env load failed path=%s", env_path)
            continue


class R2Storage:
    def __init__(self) -> None:
        _load_local_env()
        self.s3_url = _env("R2_S3_API_URL") or _env("EXPO_PUBLIC_R2_S3_API_URL")
        self.access_key = _env("R2_ACCESS_KEY_ID") or _env(
            "EXPO_PUBLIC_R2_ACCESS_KEY_ID"
        )
        self.secret_key = _env("R2_SECRET_ACCESS_KEY") or _env(
            "EXPO_PUBLIC_R2_SECRET_ACCESS_KEY"
        )
        self.raw_bucket = _env("R2_BUCKET_RAW_IMAGES") or _env(
            "EXPO_PUBLIC_R2_BUCKET_RAW_IMAGES"
        )
        self.raw_public_url = _env("R2_URL_RAW_IMAGES") or _env(
            "EXPO_PUBLIC_R2_URL_RAW_IMAGES"
        )
        self.wardrobe_bucket = _env("R2_BUCKET_WARDROBE") or _env(
            "EXPO_PUBLIC_R2_BUCKET_WARDROBE"
        )
        self.wardrobe_public_url = _env("R2_URL_WARDROBE") or _env(
            "EXPO_PUBLIC_R2_URL_WARDROBE"
        )
        self.style_boards_bucket = _env("R2_BUCKET_STYLE_BOARDS") or _env(
            "EXPO_PUBLIC_R2_BUCKET_STYLE_BOARDS"
        )
        self.style_boards_public_url = _env("R2_URL_STYLE_BOARDS") or _env(
            "EXPO_PUBLIC_R2_URL_STYLE_BOARDS"
        )

    _shared_client = None
    _shared_client_lock = threading.Lock()

    def _client(self):
        if Minio is None:
            raise R2StorageError("minio package is not installed on backend.")

        if not self.s3_url or not self.access_key or not self.secret_key:
            raise R2StorageError("Missing R2 backend configuration.")

        # Cached singleton: avoid building a new HTTP pool per request.
        cls = R2Storage
        if cls._shared_client is not None:
            return cls._shared_client

        with cls._shared_client_lock:
            if cls._shared_client is not None:
                return cls._shared_client

            endpoint = self.s3_url.replace("https://", "").replace("http://", "")
            http_client = None
            if urllib3 is not None:
                try:
                    http_client = urllib3.PoolManager(
                        num_pools=10,
                        maxsize=20,
                        retries=urllib3.Retry(
                            total=3,
                            backoff_factor=0.4,
                            status_forcelist=[500, 502, 503, 504],
                            allowed_methods=("GET", "PUT", "DELETE", "HEAD"),
                        ),
                        timeout=urllib3.Timeout(connect=2.0, read=15.0),
                    )
                except Exception:
                    logger.exception("urllib3 PoolManager init failed; using minio default")
                    http_client = None

            client_kwargs: Dict[str, object] = {
                "access_key": self.access_key,
                "secret_key": self.secret_key,
                "region": "auto",
            }
            if http_client is not None:
                client_kwargs["http_client"] = http_client
            cls._shared_client = Minio(endpoint, **client_kwargs)
            return cls._shared_client

    def upload_avatar(self, *, user_id: str, image_bytes: bytes) -> str:
        if not self.raw_bucket or not self.raw_public_url:
            raise R2StorageError("Missing raw bucket/public URL configuration.")

        file_name = f"avatar_{user_id}_{int.from_bytes(os.urandom(4), 'big')}.png"
        client = self._client()
        client.put_object(
            self.raw_bucket,
            file_name,
            BytesIO(image_bytes),
            length=len(image_bytes),
            content_type="image/png",
        )
        return f"{self.raw_public_url}/{file_name}"

    def upload_wardrobe_images(
        self,
        *,
        file_id: str,
        raw_image_bytes: bytes,
        masked_image_bytes: bytes,
    ) -> Dict[str, str]:
        if not self.raw_bucket or not self.raw_public_url:
            raise R2StorageError("Missing raw bucket/public URL configuration.")
        if not self.wardrobe_bucket or not self.wardrobe_public_url:
            raise R2StorageError("Missing wardrobe bucket/public URL configuration.")

        raw_file_name = f"raw_{file_id}.png"
        masked_file_name = f"wardrobe_{file_id}.png"
        normalized_file_name = f"wardrobe_{file_id}_normalized.png"

        # Long-term style-board quality:
        # Store a 1024x1024 transparent PNG for composition, while keeping
        # raw + masked URLs for audit/backward compatibility.
        normalized_image_bytes = masked_image_bytes
        if normalize_wardrobe_image_bytes is not None:
            try:
                normalized_image_bytes = normalize_wardrobe_image_bytes(
                    masked_image_bytes or raw_image_bytes,
                    canvas_size=1024,
                    object_fill=0.82,
                    output_format="PNG",
                )
            except Exception:
                # Never break wardrobe upload because normalization failed.
                # Fallback keeps the old masked image behavior.
                normalized_image_bytes = masked_image_bytes or raw_image_bytes

        client = self._client()

        # Upload the three variants concurrently (still IO-bound on the same
        # HTTP pool but at least overlapping the round-trip latency).
        from concurrent.futures import ThreadPoolExecutor, wait

        def _put(bucket: str, name: str, data: bytes) -> None:
            client.put_object(
                bucket,
                name,
                BytesIO(data),
                length=len(data),
                content_type="image/png",
            )

        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="r2-upload") as ex:
            futures = [
                ex.submit(_put, self.raw_bucket, raw_file_name, raw_image_bytes),
                ex.submit(
                    _put, self.wardrobe_bucket, masked_file_name, masked_image_bytes
                ),
                ex.submit(
                    _put,
                    self.wardrobe_bucket,
                    normalized_file_name,
                    normalized_image_bytes,
                ),
            ]
            wait(futures)
            for fut in futures:
                exc = fut.exception()
                if exc is not None:
                    raise exc

        normalized_image_url = f"{self.wardrobe_public_url}/{normalized_file_name}"
        masked_image_url = f"{self.wardrobe_public_url}/{masked_file_name}"

        return {
            "raw_file_name": raw_file_name,
            "masked_file_name": masked_file_name,
            "normalized_file_name": normalized_file_name,
            "raw_image_url": f"{self.raw_public_url}/{raw_file_name}",
            "masked_image_url": masked_image_url,
            "normalized_image_url": normalized_image_url,
            "normalized_url": normalized_image_url,
            # Compatibility field for newer pipelines that expect image_url to
            # already be the best display asset.
            "image_url": normalized_image_url or masked_image_url,
        }

    @staticmethod
    def catalog_object_name(item_id: str) -> str:
        """Single canonical catalog object key. Deterministic from item_id so
        the client can build the URL without a DB field."""
        return f"catalog_{str(item_id or '').strip()}.jpg"

    @staticmethod
    def catalog_png_object_name(item_id: str) -> str:
        """Canonical transparent catalog PNG key."""
        return f"catalog_{str(item_id or '').strip()}.png"

    def build_catalog_url(self, item_id: str) -> str:
        """Deterministic public catalog URL for an item, or '' if unconfigured."""
        if not self.wardrobe_public_url or not str(item_id or "").strip():
            return ""
        return f"{self.wardrobe_public_url}/{self.catalog_object_name(item_id)}"

    def build_catalog_png_url(self, item_id: str) -> str:
        """Deterministic public catalog PNG URL for an item, or '' if unconfigured."""
        if not self.wardrobe_public_url or not str(item_id or "").strip():
            return ""
        return f"{self.wardrobe_public_url}/{self.catalog_png_object_name(item_id)}"

    def upload_catalog_image(
        self,
        *,
        file_id: str,
        image_bytes: bytes,
        extension: str = "jpg",
    ) -> Dict[str, str]:
        """Upload a single catalog (clean centered product) image to the
        wardrobe bucket using the canonical deterministic key
        (catalog_{item_id}.jpg). Returns {catalog_file_name, catalog_url}."""
        if not self.wardrobe_bucket or not self.wardrobe_public_url:
            raise R2StorageError("Missing wardrobe bucket/public URL configuration.")
        # One canonical format so the client can derive the URL from item_id.
        file_name = self.catalog_object_name(file_id)
        content_type = "image/jpeg"
        del extension  # kept for signature compatibility; format is fixed to .jpg
        client = self._client()
        client.put_object(
            self.wardrobe_bucket,
            file_name,
            BytesIO(image_bytes),
            length=len(image_bytes),
            content_type=content_type,
        )
        return {
            "catalog_file_name": file_name,
            "catalog_url": f"{self.wardrobe_public_url}/{file_name}",
        }

    def upload_catalog_png(
        self,
        *,
        file_id: str,
        image_bytes: bytes,
    ) -> Dict[str, str]:
        """Upload the premium transparent catalog PNG to the wardrobe bucket."""
        if not self.wardrobe_bucket or not self.wardrobe_public_url:
            raise R2StorageError("Missing wardrobe bucket/public URL configuration.")
        file_name = self.catalog_png_object_name(file_id)
        client = self._client()
        client.put_object(
            self.wardrobe_bucket,
            file_name,
            BytesIO(image_bytes),
            length=len(image_bytes),
            content_type="image/png",
        )
        url = f"{self.wardrobe_public_url}/{file_name}"
        return {
            "catalog_png_file_name": file_name,
            "catalog_png_url": url,
            "normalized_url": url,
        }

    def upload_style_board_image(
        self,
        *,
        user_id: str,
        image_bytes: bytes,
        extension: str = "png",
    ) -> Dict[str, str]:
        # Prefer dedicated style-boards bucket if configured, otherwise fall back to raw bucket.
        target_bucket = self.style_boards_bucket or self.raw_bucket
        target_public_url = self.style_boards_public_url or self.raw_public_url
        if not target_bucket or not target_public_url:
            raise R2StorageError("Missing style board bucket/public URL configuration.")

        ext = (extension or "png").lower().strip(".")
        if ext not in ("png", "jpg", "jpeg", "webp"):
            ext = "png"

        file_name = f"style_board_{user_id}_{uuid.uuid4().hex}.{ext}"
        content_type = "image/png"
        if ext in ("jpg", "jpeg"):
            content_type = "image/jpeg"
        elif ext == "webp":
            content_type = "image/webp"

        client = self._client()
        client.put_object(
            target_bucket,
            file_name,
            BytesIO(image_bytes),
            length=len(image_bytes),
            content_type=content_type,
        )

        return {
            "file_name": file_name,
            "image_url": f"{target_public_url}/{file_name}",
            "content_type": content_type,
        }

    def read_object_bytes(
        self,
        *,
        bucket: str,
        object_name: str,
        max_bytes: int = 15 * 1024 * 1024,
    ) -> bytes:
        safe_bucket = str(bucket or "").strip()
        safe_name = str(object_name or "").strip().strip("/")
        if not safe_bucket or not safe_name:
            return b""

        client = self._client()
        response = None
        try:
            response = client.get_object(safe_bucket, safe_name)
            data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise R2StorageError(
                    f"Object too large to read safely: {safe_name} ({len(data)} bytes)"
                )
            return data
        except Exception:
            return b""
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
                try:
                    response.release_conn()
                except Exception:
                    pass

    def delete_wardrobe_images(
        self,
        *,
        raw_file_name: str = "",
        masked_file_name: str = "",
        normalized_file_name: str = "",
    ) -> Dict[str, bool]:
        client = self._client()
        result = {
            "raw_deleted": False,
            "masked_deleted": False,
            "normalized_deleted": False,
        }

        if raw_file_name and self.raw_bucket:
            try:
                client.remove_object(self.raw_bucket, raw_file_name)
                result["raw_deleted"] = True
            except Exception:
                # Keep delete idempotent even if object is already missing.
                pass

        if masked_file_name and self.wardrobe_bucket:
            try:
                client.remove_object(self.wardrobe_bucket, masked_file_name)
                result["masked_deleted"] = True
            except Exception:
                # Keep delete idempotent even if object is already missing.
                pass

        if normalized_file_name and self.wardrobe_bucket:
            try:
                client.remove_object(self.wardrobe_bucket, normalized_file_name)
                result["normalized_deleted"] = True
            except Exception:
                # Keep delete idempotent even if object is already missing.
                pass

        return result
