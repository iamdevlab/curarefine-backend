# app/services/redis_client.py

import os
import redis
import json
from app.services.data_cleaner import to_python_type

# ----------------------------
# Environment Variables
# ----------------------------
REDIS_URL = os.getenv("REDIS_URL")
DEFAULT_EXPIRE = 3600  # 1 hour

# ----------------------------
# Lazy Redis Client
# ----------------------------
redis_client = None  # lazy initialization


def get_redis_client():
    """Return Redis client, initialize if needed."""
    global redis_client
    if redis_client is None:
        if not REDIS_URL:
            print("[Redis] REDIS_URL not set, skipping Redis")
            return None
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            redis_client.ping()
            print("[Redis] Connected successfully")
        except redis.ConnectionError as e:
            print(f"[Redis] Connection failed: {e}")
            redis_client = None
    return redis_client


# ----------------------------
# Legacy init_redis() for compatibility
# ----------------------------
def init_redis():
    """Legacy startup function - initialize Redis and test connection."""
    client = get_redis_client()
    if client:
        print("[Redis] Initialized successfully")
    else:
        print("[Redis] Not available")


# ----------------------------
# Session Helpers
# ----------------------------
def set_session(
    user_id: int, file_id: str, session_data: dict, expire_seconds=DEFAULT_EXPIRE
):
    client = get_redis_client()
    if not client:
        return
    key = f"session:{user_id}:{file_id}"
    client.set(key, json.dumps(session_data, default=to_python_type), ex=expire_seconds)


def get_session(user_id: int, file_id: str):
    client = get_redis_client()
    if not client:
        return None
    key = f"session:{user_id}:{file_id}"
    data = client.get(key)
    return json.loads(data) if data else None


def delete_session(user_id: int, file_id: str):
    client = get_redis_client()
    if not client:
        return
    key = f"session:{user_id}:{file_id}"
    client.delete(key)


def session_exists(user_id: int, file_id: str) -> bool:
    client = get_redis_client()
    if not client:
        return False
    key = f"session:{user_id}:{file_id}"
    return client.exists(key) == 1
