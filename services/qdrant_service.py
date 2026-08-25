import logging
import os
import time
import uuid
from dotenv import load_dotenv

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        PointStruct,
        Distance,
        VectorParams,
        Filter,
        FieldCondition,
        MatchValue,
    )
    HAS_QDRANT_CLIENT = True
except ImportError:  # pragma: no cover
    HAS_QDRANT_CLIENT = False
    QdrantClient = None
    PointStruct = Distance = VectorParams = Filter = FieldCondition = MatchValue = None

from services.image_fingerprint import hamming_distance_hex

load_dotenv()

logger = logging.getLogger(__name__)


class QdrantService:

    def __init__(self):
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.timeout_seconds = float(
            os.getenv("QDRANT_TIMEOUT_SECONDS", "2.5") or "2.5"
        )

        self.collection = os.getenv("QDRANT_COLLECTION", "wardrobe")
        self.image_collection = os.getenv("QDRANT_IMAGE_COLLECTION", "wardrobe_images")
        self.memory_collection = os.getenv("QDRANT_MEMORY_COLLECTION", "outfit_memory")
        self.user_memory_collection = os.getenv(
            "QDRANT_USER_MEMORY_COLLECTION", "user_memory"
        )

        self.vector_size = 512
        self.memory_vector_size = int(os.getenv("QDRANT_MEMORY_VECTOR_SIZE", "8"))
        self.user_memory_vector_size = self.vector_size

        self.client = None
        self._initialized = False

        if self.url and HAS_QDRANT_CLIENT:
            try:
                # Critical: enforce low default timeouts so app startup never hangs
                # when Qdrant is misconfigured or unreachable.
                self.client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=self.timeout_seconds,
                )
            except Exception:
                logger.exception("Qdrant client init failed")

    # =========================
    # INIT
    # =========================
    def init(self):
        if not self.client or self._initialized:
            return

        logger.info("Initializing Qdrant...")

        self._create_collection(self.collection, self.vector_size)
        self._create_collection(self.image_collection, self.vector_size)
        self._create_collection(self.memory_collection, self.memory_vector_size)
        self._create_collection(
            self.user_memory_collection, self.user_memory_vector_size
        )

        self._initialized = True

    def _create_collection(self, name, size):
        try:
            existing = [c.name for c in self.client.get_collections().collections]

            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=size, distance=Distance.COSINE),
                )
                logger.info("Created Qdrant collection: %s", name)

        except Exception:
            logger.exception("Qdrant collection init error for %s", name)

    def _ensure(self):
        if not self.client:
            return False
        if not self._initialized:
            self.init()
        return True

    def enabled(self):
        return bool(self.client)

    # =========================
    # 🔥 NORMALIZATION
    # =========================
    def _normalize(self, vector):
        if not vector:
            return vector

        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0:
            return vector

        return [v / norm for v in vector]

    def _fit_vector(self, vector, size=None):
        if not vector:
            return []
        size = int(size or self.vector_size)
        fitted = [float(v) for v in vector]
        if len(fitted) == size:
            return fitted
        if len(fitted) < size:
            return fitted + [0.0] * (size - len(fitted))
        return fitted[:size]

    # =========================
    # UPSERT (MAIN)
    # =========================
    def upsert_item(self, item_id, vector, payload):
        if not self._ensure():
            return

        vector = self._fit_vector(vector)
        if len(vector) != self.vector_size:
            logger.warning(
                "Qdrant vector size mismatch: got %s, expected %s",
                len(vector) if vector else 0,
                self.vector_size,
            )
            return

        vector = self._normalize(vector)

        payload["timestamp"] = time.time()

        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(
                        id=item_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
        except Exception:
            logger.exception("Qdrant upsert failed")

    # =========================
    # 🔥 BATCH UPSERT (NEW)
    # =========================
    def upsert_batch(self, items):
        if not self._ensure():
            return

        points = []

        for item in items:
            vec = self._fit_vector(item.get("vector"))
            if not vec or len(vec) != self.vector_size:
                continue

            vec = self._normalize(vec)

            payload = item.get("payload", {})
            payload["timestamp"] = time.time()

            points.append(
                PointStruct(
                    id=item.get("id", str(uuid.uuid4())),
                    vector=vec,
                    payload=payload,
                )
            )

        if not points:
            return

        try:
            self.client.upsert(
                collection_name=self.collection,
                points=points,
            )
        except Exception:
            logger.exception("Qdrant batch upsert failed")

    # =========================
    # 🔥 SEARCH (WITH THRESHOLD)
    # =========================
    def search_similar(
        self, vector, user_id, category=None, limit=5, score_threshold=0.6
    ):
        if not self._ensure() or not vector:
            return []

        vector = self._normalize(vector)

        try:
            must = [FieldCondition(key="userId", match=MatchValue(value=str(user_id)))]

            if category:
                must.append(
                    FieldCondition(key="category", match=MatchValue(value=category))
                )

            query_filter = Filter(must=must)

            results = self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=limit,
                query_filter=query_filter,
            )

            return [
                {
                    "id": str(r.id),
                    "score": float(r.score),
                    "payload": r.payload or {},
                }
                for r in results
                if r.score >= score_threshold
            ]

        except Exception:
            logger.exception("Qdrant search failed")
            return []

    def _search_collection(
        self, collection_name, vector, user_id, limit=5, score_threshold=0.6
    ):
        if not self._ensure() or not vector:
            return []

        vector = self._fit_vector(vector)
        if len(vector) != self.vector_size:
            logger.warning(
                "Qdrant vector size mismatch: got %s, expected %s",
                len(vector),
                self.vector_size,
            )
            return []

        vector = self._normalize(vector)

        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="userId", match=MatchValue(value=str(user_id))
                        )
                    ]
                ),
            )
            return [
                {
                    "id": str(r.id),
                    "score": float(r.score),
                    "payload": r.payload or {},
                }
                for r in results
                if r.score >= score_threshold
            ]
        except Exception:
            logger.exception(
                "Qdrant duplicate search failed collection=%s", collection_name
            )
            return []

    def find_duplicate(self, vector, user_id, threshold=0.97):
        if not vector:
            return {
                "checked": False,
                "is_duplicate": False,
                "reason": None,
                "confidence": 0.0,
                "matched_item_id": None,
                "score": 0.0,
            }
        matches = self._search_collection(
            self.collection,
            vector,
            user_id,
            limit=1,
            score_threshold=float(threshold),
        )
        if not matches:
            return {
                "checked": True,
                "is_duplicate": False,
                "reason": None,
                "confidence": 0.0,
                "matched_item_id": None,
                "score": 0.0,
            }
        top = matches[0]
        score = float(top.get("score") or 0.0)
        return {
            "checked": True,
            "is_duplicate": True,
            "reason": "metadata",
            "confidence": score,
            "matched_item_id": top.get("id"),
            "id": top.get("id"),
            "score": score,
            "payload": top.get("payload") or {},
        }

    def find_image_duplicate(self, vector, user_id, threshold=0.985):
        if not vector:
            return {
                "checked": False,
                "is_duplicate": False,
                "reason": None,
                "confidence": 0.0,
                "matched_item_id": None,
                "score": 0.0,
            }
        matches = self._search_collection(
            self.image_collection,
            vector,
            user_id,
            limit=1,
            score_threshold=float(threshold),
        )
        if not matches:
            return {
                "checked": True,
                "is_duplicate": False,
                "reason": None,
                "confidence": 0.0,
                "matched_item_id": None,
                "score": 0.0,
            }
        top = matches[0]
        score = float(top.get("score") or 0.0)
        return {
            "checked": True,
            "is_duplicate": True,
            "reason": "image_vector",
            "confidence": score,
            "matched_item_id": top.get("id"),
            "id": top.get("id"),
            "score": score,
            "payload": top.get("payload") or {},
        }

    def upsert_image_vector(self, item_id, vector, payload):
        if not self._ensure():
            return

        vector = self._fit_vector(vector)
        if not vector or len(vector) != self.vector_size:
            logger.warning(
                "Qdrant image vector size mismatch: got %s, expected %s",
                len(vector) if vector else 0,
                self.vector_size,
            )
            return

        vector = self._normalize(vector)
        payload = dict(payload or {})
        payload["timestamp"] = time.time()

        try:
            self.client.upsert(
                collection_name=self.image_collection,
                points=[
                    PointStruct(
                        id=item_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
        except Exception:
            logger.exception("Qdrant image upsert failed")

    def upsert_wardrobe_item(self, item):
        item = dict(item or {})
        item_id = str(item.get("id") or "").strip()
        vector = item.get("embedding") or item.get("vector") or []
        vector = self._fit_vector(vector)
        if not item_id or not vector or len(vector) != self.vector_size:
            return

        payload = {
            "userId": item.get("userId"),
            "type": item.get("type"),
            "category": item.get("category"),
            "color": item.get("color"),
            "image_url": item.get("image_url"),
            "pixel_hash": item.get("pixel_hash"),
            "qdrant_point_id": item_id,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        self.upsert_item(item_id, vector, payload)

    def delete_item(self, item_id):
        if not self._ensure() or not item_id:
            return
        for collection_name in (self.collection, self.image_collection):
            try:
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=[item_id],
                )
            except Exception:
                logger.exception("Qdrant delete failed collection=%s", collection_name)

    # =========================
    # 🔥 FAST PIXEL DUPLICATE
    # =========================
    def find_pixel_duplicate(self, user_id, pixel_hash, max_distance=6):
        if not self._ensure() or not pixel_hash:
            return {
                "checked": False,
                "is_duplicate": False,
                "reason": None,
                "confidence": 0.0,
                "matched_item_id": None,
            }

        try:
            # 🔥 only fetch limited candidates
            response = self.client.scroll(
                collection_name=self.collection,
                limit=100,
                with_payload=True,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="userId", match=MatchValue(value=str(user_id))
                        )
                    ]
                ),
            )

            points = response[0] if isinstance(response, tuple) else []

            for point in points:
                candidate_hash = (point.payload or {}).get("pixel_hash")
                if not candidate_hash:
                    continue

                dist = hamming_distance_hex(pixel_hash, candidate_hash)

                if dist is not None and dist <= max_distance:
                    confidence = max(0.0, 1.0 - (float(dist) / max(float(max_distance), 1.0)))
                    return {
                        "checked": True,
                        "is_duplicate": True,
                        "reason": "pixel_hash",
                        "confidence": confidence,
                        "matched_item_id": str(point.id),
                        "id": str(point.id),
                        "distance": dist,
                        "payload": point.payload or {},
                    }

            return {
                "checked": True,
                "is_duplicate": False,
                "reason": None,
                "confidence": 0.0,
                "matched_item_id": None,
            }

        except Exception:
            return {
                "checked": True,
                "is_duplicate": False,
                "reason": None,
                "confidence": 0.0,
                "matched_item_id": None,
            }

    # =========================
    # USER MEMORY
    # =========================
    def upsert_user_memory(self, user_id, vector, payload):
        if not self._ensure() or not vector:
            return None

        vector = self._normalize(vector)

        try:
            self.client.upsert(
                collection_name=self.user_memory_collection,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
        except Exception:
            logger.exception("Qdrant user memory upsert failed")

    # =========================
    # STATUS
    # =========================
    def status(self):
        return {
            "enabled": self.enabled(),
            "initialized": self._initialized,
            "collection": self.collection,
            "image_collection": self.image_collection,
            "vector_size": self.vector_size,
        }


# =========================
# SINGLETON
# =========================
qdrant_service = QdrantService()
