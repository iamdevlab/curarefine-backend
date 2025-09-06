# app/services/postgres_client.py

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import HTTPException

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from app.services.encryption_service import encrypt, decrypt

# --- Connection Pool ---
connection_pool = None


def init_db_pool():
    """Initializes the database connection pool. Should be called on startup."""
    global connection_pool
    if connection_pool is None:
        try:
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL environment variable not set.")
            connection_pool = pool.SimpleConnectionPool(
                minconn=1, maxconn=10, dsn=db_url
            )
            print("[Postgres] Connection pool created successfully.")
        except Exception as e:
            print(f"[Postgres] Failed to create connection pool: {e}")
            connection_pool = None


# --- Standardized Connection Helpers ---


def get_connection():
    """Gets a connection from the pool."""
    if not connection_pool:
        raise Exception("Postgres connection pool is not available.")
    return connection_pool.getconn()


def put_connection(conn):
    """Returns a connection to the pool."""
    if connection_pool:
        connection_pool.putconn(conn)


# --- FastAPI Dependency ---
def get_db_cursor():
    """FastAPI dependency that provides a transaction-managed database cursor."""
    if not connection_pool:
        raise HTTPException(status_code=503, detail="Database connection not available")

    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        yield cursor
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Database transaction failed: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        if conn:
            put_connection(conn)


# ----------------------------
# Schema Initialization
# ----------------------------
def init_db() -> None:
    """Create tables if they do not exist."""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            schema_queries = [
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    hashed_password VARCHAR(255) NOT NULL,
                    full_name VARCHAR(100),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """,
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id SERIAL PRIMARY KEY,
                    user_id INT NOT NULL,
                    file_id VARCHAR(255) UNIQUE NOT NULL,
                    project_name VARCHAR(255) NOT NULL,
                    upload_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    status VARCHAR(50) NOT NULL DEFAULT 'Pending',
                    row_count INT,
                    file_size INT,
                    missing_values_count INT DEFAULT 0, 
                    outlier_count INT DEFAULT 0         
                );
                """,
                """
                CREATE TABLE IF NOT EXISTS user_llm_settings (
                    id SERIAL PRIMARY KEY,
                    user_id INT NOT NULL,
                    provider VARCHAR(50) NOT NULL,
                    model_id VARCHAR(100),
                    api_key BYTEA,
                    endpoint_url VARCHAR(255),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    CONSTRAINT user_llm_settings_user_provider_key UNIQUE (user_id, provider)
                );
                """,
                """
                CREATE TABLE IF NOT EXISTS cleaning_sessions (
                    id SERIAL PRIMARY KEY,
                    user_id INT NOT NULL,
                    file_id VARCHAR(255) NOT NULL,
                    session_data JSONB,
                    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(user_id, file_id)
                );
                """,
            ]
            for query in schema_queries:
                cursor.execute(query)
        conn.commit()
        print("[Postgres] Schema checked/initialized successfully.")
    except Exception as e:
        print(f"[Postgres] Error in init_db: {e}")
    finally:
        if conn:
            put_connection(conn)


async def init_db_async():
    """Async wrapper to initialize pool and schema on startup."""
    from starlette.concurrency import run_in_threadpool

    await run_in_threadpool(init_db_pool)
    await run_in_threadpool(init_db)


# --- Helper Functions (Copied from your original file) ---


def get_user_by_id(user_id: int, cursor: RealDictCursor):
    """Fetch a single user by their ID."""
    cursor.execute(
        "SELECT id, username, full_name, email FROM users WHERE id = %s", (user_id,)
    )
    return cursor.fetchone()


def get_user_by_username(
    username: str, cursor: RealDictCursor
) -> Optional[Dict[str, Any]]:
    """Fetch a single user by their username."""
    cursor.execute("SELECT * FROM users WHERE username = %s;", (username,))
    return cursor.fetchone()


def create_session(user_id: int, file_id: str, session_data: Dict[str, Any]) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO cleaning_sessions (user_id, file_id, session_data, last_updated)
                VALUES (%s, %s, %s, %s) ON CONFLICT (user_id, file_id)
                DO UPDATE SET session_data = EXCLUDED.session_data, last_updated = EXCLUDED.last_updated;
                """,
                (user_id, file_id, json.dumps(session_data), datetime.now()),
            )
        conn.commit()
    finally:
        if conn:
            put_connection(conn)


def get_session(user_id: int, file_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT session_data FROM cleaning_sessions WHERE user_id = %s AND file_id = %s;",
                (user_id, file_id),
            )
            result = cursor.fetchone()
            return result["session_data"] if result else None
    finally:
        if conn:
            put_connection(conn)


def delete_session(user_id: int, file_id: str) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM cleaning_sessions WHERE user_id = %s AND file_id = %s;",
                (user_id, file_id),
            )
        conn.commit()
    finally:
        if conn:
            put_connection(conn)


def create_project_entry(project_data: Dict[str, Any], cursor: RealDictCursor):
    query = """
        INSERT INTO projects
            (user_id, file_id, project_name, row_count, file_size, missing_values_count, outlier_count, status)
        VALUES
            (%(user_id)s, %(file_id)s, %(project_name)s, %(row_count)s, %(file_size)s, %(missing_values_count)s, %(outlier_count)s, 'Pending');
        """
    cursor.execute(query, project_data)


def update_project_status(
    user_id: int, file_id: str, new_status: str, cursor: RealDictCursor
):
    query = "UPDATE projects SET status = %s WHERE user_id = %s AND file_id = %s;"
    cursor.execute(query, (new_status, user_id, file_id))


def save_llm_settings(user_id: int, settings: dict, cursor: RealDictCursor):
    provider = settings.get("provider")
    encrypted_api_key = encrypt(settings.get("api_key"))
    model_id = settings.get("model_id")
    endpoint_url = settings.get("endpoint_url")
    cursor.execute(
        """
        INSERT INTO user_llm_settings (user_id, provider, api_key, model_id, endpoint_url, updated_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        ON CONFLICT (user_id, provider)
        DO UPDATE SET api_key = EXCLUDED.api_key, model_id = EXCLUDED.model_id,
                      endpoint_url = EXCLUDED.endpoint_url, updated_at = NOW();
        """,
        (user_id, provider, encrypted_api_key, model_id, endpoint_url),
    )


def get_all_llm_settings_for_user(user_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT provider, model_id FROM user_llm_settings WHERE user_id = %s;",
                (user_id,),
            )
            return cursor.fetchall()
    finally:
        if conn:
            put_connection(conn)


def get_active_llm_settings_for_user(
    user_id: int, cursor: RealDictCursor
) -> Optional[Dict[str, Any]]:
    query = """
        SELECT s.provider, s.api_key, s.model_id,
               COALESCE(s.endpoint_url, p.default_endpoint_url) AS final_endpoint_url
        FROM user_llm_settings s
        LEFT JOIN provider_endpoint p ON s.provider = p.identifier
        WHERE s.user_id = %s ORDER BY s.updated_at DESC LIMIT 1;
        """
    cursor.execute(query, (user_id,))
    settings = cursor.fetchone()
    if settings:
        settings = dict(settings)
        if settings.get("api_key"):
            settings["api_key"] = decrypt(settings["api_key"])
        return settings
    return None


def get_projects_for_user(user_id: int, cursor: RealDictCursor) -> List[Dict[str, Any]]:
    query = """
        SELECT id, file_id, project_name, status, row_count, upload_time 
        FROM projects WHERE user_id = %s ORDER BY upload_time DESC;
    """
    cursor.execute(query, (user_id,))
    return cursor.fetchall()


def get_dashboard_insights(user_id: int, cursor: RealDictCursor) -> Dict[str, Any]:
    query_missing = """
        SELECT COUNT(*) FROM projects
        WHERE user_id = %s AND missing_values_count > 0
        AND upload_time >= NOW() - INTERVAL '7 days';
    """
    cursor.execute(query_missing, (user_id,))
    missing_values_count = cursor.fetchone()["count"]

    cursor.execute(
        "SELECT COUNT(*) FROM projects WHERE user_id = %s AND outlier_count > 0 AND upload_time >= NOW() - INTERVAL '7 days';",
        (user_id,),
    )
    outliers_detected_count = cursor.fetchone()["count"]

    cursor.execute(
        "SELECT AVG(row_count) as avg_size FROM projects WHERE user_id = %s;",
        (user_id,),
    )
    avg_result = cursor.fetchone()["avg_size"]
    average_dataset_size = int(avg_result) if avg_result is not None else 0

    return {
        "missing_values_found_in_datasets": missing_values_count,
        "outliers_detected_in_datasets": outliers_detected_count,
        "average_dataset_size_in_rows": average_dataset_size,
    }
