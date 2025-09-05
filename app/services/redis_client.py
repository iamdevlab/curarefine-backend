# app/db/redis_client.py

import redis
import os
import json
from app.services.data_cleaner import to_python_type

REDIS_URL = os.getenv("REDIS_URL")
# --------------------
# Redis Configuration
# --------------------
REDIS_HOST = os.getenv("REDIS_HOST", REDIS_URL)
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
DEFAULT_EXPIRE = 3600  # seconds, 1 hour

# --------------------
# Redis Client
# --------------------
redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
)


# --------------------
# Initialization
# --------------------
def init_redis():
    """Check Redis connection on startup"""
    try:
        redis_client.ping()
        print(f"[Redis] Connected at {REDIS_HOST}:{REDIS_PORT}, DB {REDIS_DB}")
    except redis.ConnectionError:
        print("[Redis] Connection failed! Make sure the Redis server is running.")


# --------------------
# Session Helpers
# --------------------
def set_session(
    user_id: int, file_id: str, session_data: dict, expire_seconds: int = DEFAULT_EXPIRE
):
    """Store session in Redis with expiration"""
    key = f"session:{user_id}:{file_id}"
    redis_client.set(
        key, json.dumps(session_data, default=to_python_type), ex=expire_seconds
    )


def get_session(user_id: int, file_id: str) -> dict:
    """Retrieve session from Redis"""
    key = f"session:{user_id}:{file_id}"
    data = redis_client.get(key)
    return json.loads(data) if data else None


def delete_session(user_id: int, file_id: str):
    """Delete session from Redis"""
    key = f"session:{user_id}:{file_id}"
    redis_client.delete(key)


def session_exists(user_id: int, file_id: str) -> bool:
    """Check if a session exists in Redis"""
    key = f"session:{user_id}:{file_id}"
    return redis_client.exists(key) == 1
