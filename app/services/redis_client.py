# app/services/redis_client.py

import os
import redis
import json
from app.services.data_cleaner import to_python_type

# ----------------------------
# Redis Configuration
# ----------------------------
REDIS_URL = os.getenv("REDIS_URL")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
DEFAULT_EXPIRE = 3600  # seconds, 1 hour

# ----------------------------
# Redis Client (Lazy/Hybrid)
# ----------------------------
redis_client = None  # Lazy init


def get_redis_client():
    global redis_client
    if redis_client is None:
        try:
            if REDIS_URL:
                redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            else:
                redis_client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    decode_responses=True,
                    ssl=True,
                )
            redis_client.ping()
            print(f"[Redis] Connected at {REDIS_HOST}:{REDIS_PORT}, DB {REDIS_DB}")
        except redis.ConnectionError as e:
            print(f"[Redis] Connection failed: {e}")
            redis_client = None
    return redis_client


# ----------------------------
# Legacy init_redis() for compatibility
# ----------------------------
def init_redis():
    """Check Redis connection on startup"""
    client = get_redis_client()
    if client:
        print("[Redis] Ready to use")
    else:
        print("[Redis] Not available, skipping Redis operations")


# ----------------------------
# Session Helpers
# ----------------------------
def set_session(
    user_id: int, file_id: str, session_data: dict, expire_seconds=DEFAULT_EXPIRE
):
    """Store session in Redis with expiration"""
    client = get_redis_client()
    if not client:
        return
    key = f"session:{user_id}:{file_id}"
    client.set(key, json.dumps(session_data, default=to_python_type), ex=expire_seconds)


def get_session(user_id: int, file_id: str):
    """Retrieve session from Redis"""
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
