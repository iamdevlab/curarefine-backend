# # app/services/redis_client.py
# app/services/redis_client.py

import os
import json
from upstash_redis import Redis  # Import from the correct library
from app.services.data_cleaner import to_python_type

# ----------------------------
# Upstash Redis Configuration
# ----------------------------
UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")
DEFAULT_EXPIRE = 3600  # seconds, 1 hour

# ----------------------------
# Redis Client (Lazy Init)
# ----------------------------
redis_client = None


def get_redis_client():
    global redis_client
    if redis_client is None:
        try:
            # Check if both Upstash environment variables are set
            if UPSTASH_URL and UPSTASH_TOKEN:
                # Initialize the Upstash Redis client
                redis_client = Redis(url=UPSTASH_URL, token=UPSTASH_TOKEN)
                print("[Redis] Upstash client configured and ready.")
            else:
                print("[Redis] Upstash environment variables not found.")
                # We explicitly set client to None if config is missing
                redis_client = None

        except Exception as e:
            print(f"[Redis] Failed to initialize Upstash client: {e}")
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
# Session Helpers (No changes needed here)
# ----------------------------
def set_session(
    user_id: int, file_id: str, session_data: dict, expire_seconds=DEFAULT_EXPIRE
):
    """Store session in Redis with expiration"""
    client = get_redis_client()
    if not client:
        return
    key = f"session:{user_id}:{file_id}"
    # The upstash-redis library uses the same method names
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
    return client.exists(key) > 0


# import os
# import redis
# import json
# from app.services.data_cleaner import to_python_type

# # ----------------------------
# # Redis Configuration
# # ----------------------------
# REDIS_URL = os.getenv("REDIS_URL")
# REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
# REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
# REDIS_DB = int(os.getenv("REDIS_DB", 0))
# DEFAULT_EXPIRE = 3600  # seconds, 1 hour

# # ----------------------------
# # Redis Client (Lazy/Hybrid)
# # ----------------------------
# redis_client = None  # Lazy init


# def get_redis_client():
#     global redis_client
#     if redis_client is None:
#         try:
#             if REDIS_URL:
#                 redis_client = redis.from_url(REDIS_URL, decode_responses=True)
#             else:
#                 redis_client = redis.Redis(
#                     host=REDIS_HOST,
#                     port=REDIS_PORT,
#                     db=REDIS_DB,
#                     decode_responses=True,
#                     ssl=True,
#                 )
#             redis_client.ping()
#             print(f"[Redis] Connected at {REDIS_HOST}:{REDIS_PORT}, DB {REDIS_DB}")
#         except redis.ConnectionError as e:
#             print(f"[Redis] Connection failed: {e}")
#             redis_client = None
#     return redis_client


# # ----------------------------
# # Legacy init_redis() for compatibility
# # ----------------------------
# def init_redis():
#     """Check Redis connection on startup"""
#     client = get_redis_client()
#     if client:
#         print("[Redis] Ready to use")
#     else:
#         print("[Redis] Not available, skipping Redis operations")


# # ----------------------------
# # Session Helpers
# # ----------------------------
# def set_session(
#     user_id: int, file_id: str, session_data: dict, expire_seconds=DEFAULT_EXPIRE
# ):
#     """Store session in Redis with expiration"""
#     client = get_redis_client()
#     if not client:
#         return
#     key = f"session:{user_id}:{file_id}"
#     client.set(key, json.dumps(session_data, default=to_python_type), ex=expire_seconds)


# def get_session(user_id: int, file_id: str):
#     """Retrieve session from Redis"""
#     client = get_redis_client()
#     if not client:
#         return None
#     key = f"session:{user_id}:{file_id}"
#     data = client.get(key)
#     return json.loads(data) if data else None


# def delete_session(user_id: int, file_id: str):
#     client = get_redis_client()
#     if not client:
#         return
#     key = f"session:{user_id}:{file_id}"
#     client.delete(key)


# def session_exists(user_id: int, file_id: str) -> bool:
#     client = get_redis_client()
#     if not client:
#         return False
#     key = f"session:{user_id}:{file_id}"
#     return client.exists(key) == 1
