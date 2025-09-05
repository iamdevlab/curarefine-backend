# postgres_client.py
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Optional, Dict, Any, List
from psycopg2 import pool
from app.services.encryption_service import encrypt, decrypt

# ==== Database Config ====
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "datacura")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "hydrogen")


# ==== Connection Pool ====
# This pool is created once and shared by the application.
try:
    connection_pool = pool.SimpleConnectionPool(
        1,  # minconn
        20,  # maxconn
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
except psycopg2.OperationalError as e:
    print(f"Error creating connection pool: {e}")
    connection_pool = None


# ==== Connection Helpers ====
def get_connection():
    """Gets a connection from the pool."""
    if connection_pool:
        return connection_pool.getconn()
    raise Exception("Connection pool is not available.")


def release_connection(conn):
    """Returns a connection to the pool."""
    if connection_pool:
        connection_pool.putconn(conn)


# ==== Schema Init ====
def init_db() -> None:
    """Create tables if they do not exist."""
    # This function uses a single connection to run all schema queries.
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            # (Schema queries are omitted for brevity, but belong here)
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
                # ... include all your other CREATE TABLE statements here ...
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
            ]
            for query in schema_queries:
                cursor.execute(query)
            conn.commit()
    finally:
        if conn:
            release_connection(conn)


# ==== User Management ====
# This function takes a cursor, so it's used within another transaction. No change needed.
def get_user_by_username(
    username: str,
    cursor: RealDictCursor,
) -> Optional[Dict[str, Any]]:
    query = "SELECT id, username, hashed_password FROM users WHERE username = %s;"
    cursor.execute(query, (username,))
    return cursor.fetchone()


# ==== Cleaning Sessions ====
def create_session(user_id: int, file_id: str, session_data: Dict[str, Any]) -> None:
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            query = """
                INSERT INTO cleaning_sessions (user_id, file_id, session_data, last_updated)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, file_id)
                DO UPDATE SET session_data = EXCLUDED.session_data,
                              last_updated = EXCLUDED.last_updated;
            """
            cursor.execute(
                query, (user_id, file_id, json.dumps(session_data), datetime.now())
            )
        conn.commit()
    finally:
        if conn:
            release_connection(conn)


def get_session(user_id: int, file_id: str) -> Optional[Dict[str, Any]]:
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            query = "SELECT session_data FROM cleaning_sessions WHERE user_id = %s AND file_id = %s;"
            cursor.execute(query, (user_id, file_id))
            result = cursor.fetchone()
            return result[0] if result else None
    finally:
        if conn:
            release_connection(conn)


def delete_session(user_id: int, file_id: str) -> None:
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            query = "DELETE FROM cleaning_sessions WHERE user_id = %s AND file_id = %s;"
            cursor.execute(query, (user_id, file_id))
        conn.commit()
    finally:
        if conn:
            release_connection(conn)


# ==== Functions that take a cursor (no changes needed) ====
# These are helper functions intended to be part of a larger transaction in an API route.
def create_project_entry(project_data: Dict[str, Any], cursor):
    query = """
        INSERT INTO projects
            (user_id, file_id, project_name, row_count, file_size, missing_values_count, outlier_count, status)
        VALUES
            (%(user_id)s, %(file_id)s, %(project_name)s, %(row_count)s, %(file_size)s, %(missing_values_count)s, %(outlier_count)s, 'Pending');
    """
    cursor.execute(query, project_data)


def save_llm_settings(user_id: int, settings: dict, cursor):
    provider = settings.get("provider")
    encrypted_api_key = encrypt(settings.get("api_key"))
    model_id = settings.get("model_id")
    endpoint_url = settings.get("endpoint_url")
    query = """
        INSERT INTO user_llm_settings (user_id, provider, api_key, model_id, endpoint_url, updated_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        ON CONFLICT (user_id, provider)
        DO UPDATE SET api_key = EXCLUDED.api_key, model_id = EXCLUDED.model_id,
                      endpoint_url = EXCLUDED.endpoint_url, updated_at = NOW();
    """
    cursor.execute(
        query, (user_id, provider, encrypted_api_key, model_id, endpoint_url)
    )


def update_project_status(user_id: int, file_id: str, new_status: str, cursor):
    query = "UPDATE projects SET status = %s WHERE user_id = %s AND file_id = %s;"
    cursor.execute(query, (new_status, user_id, file_id))


# ==== Functions that manage their own connection ====
def get_all_llm_settings_for_user(user_id: int) -> List[Dict[str, Any]]:
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = (
                "SELECT provider, model_id FROM user_llm_settings WHERE user_id = %s;"
            )
            cursor.execute(query, (user_id,))
            return cursor.fetchall()
    finally:
        if conn:
            release_connection(conn)


def get_active_llm_settings_for_user(
    user_id: int, cursor: RealDictCursor
) -> Optional[Dict[str, Any]]:
    # This function is intended to be used within a transaction, so it takes a cursor. No change needed.
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
        encrypted_api_key = settings.get("api_key")
        if encrypted_api_key:
            settings["api_key"] = decrypt(encrypted_api_key)
        return settings
    return None


def get_projects_for_user(user_id: int, cursor) -> List[Dict[str, Any]]:
    # This function is intended to be used within a transaction, so it takes a cursor. No change needed.
    query = """
        SELECT id, file_id, project_name, status, row_count, upload_time 
        FROM projects WHERE user_id = %s ORDER BY upload_time DESC;
    """
    cursor.execute(query, (user_id,))
    return cursor.fetchall()


def get_dashboard_insights(user_id: int, cursor) -> Dict[str, Any]:
    # This function is intended to be used within a transaction, so it takes a cursor. No change needed.
    query_missing = """
        SELECT COUNT(*) FROM projects
        WHERE user_id = %s AND missing_values_count > 0
        AND upload_time >= NOW() - INTERVAL '7 days';
    """
    cursor.execute(query_missing, (user_id,))
    missing_values_count = cursor.fetchone()["count"]

    query_outliers = """
        SELECT COUNT(*) FROM projects
        WHERE user_id = %s AND outlier_count > 0
        AND upload_time >= NOW() - INTERVAL '7 days';
    """
    cursor.execute(query_outliers, (user_id,))
    outliers_detected_count = cursor.fetchone()["count"]

    query_avg_size = (
        "SELECT AVG(row_count) as avg_size FROM projects WHERE user_id = %s;"
    )
    cursor.execute(query_avg_size, (user_id,))
    avg_result = cursor.fetchone()["avg_size"]
    average_dataset_size = int(avg_result) if avg_result is not None else 0

    return {
        "missing_values_found_in_datasets": missing_values_count,
        "outliers_detected_in_datasets": outliers_detected_count,
        "average_dataset_size_in_rows": average_dataset_size,
    }
