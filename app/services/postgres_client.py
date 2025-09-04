# postgres_client.py
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Optional, Dict, Any, List
from .encryption_service import encrypt, decrypt

# ==== Database Config ====
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "datacura")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "hydrogen")


# ==== Connection Helper ====
def get_connection():
    """Return a new PostgreSQL connection."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


# ==== Schema Init ====
def init_db() -> None:
    """Create tables if they do not exist."""
    schema_queries = [
        # --- Table for storing user accounts ---
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
        # --- Tables for cleaning history and state ---
        """
        CREATE TABLE IF NOT EXISTS cleaning_logs (
            id SERIAL PRIMARY KEY,
            user_id INT NOT NULL,
            file_id VARCHAR(255) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            action VARCHAR(255),
            details JSONB
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS cleaning_sessions (
            user_id INT NOT NULL,
            file_id VARCHAR(255) NOT NULL,
            session_data JSONB,
            last_updated TIMESTAMP,
            PRIMARY KEY (user_id, file_id)
        );
        """,
        # --- Table for user-specific LLM settings ---
        """
       CREATE TABLE IF NOT EXISTS user_llm_settings (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    provider VARCHAR(50) NOT NULL,
    model_id VARCHAR(100),
    api_key BYTEA,  -- CHANGED: from VARCHAR(255) to BYTEA
    endpoint_url VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """,
        # --- NEW: Table for tracking projects ---
        """
        CREATE TABLE IF NOT EXISTS projects (
            id SERIAL PRIMARY KEY,
            user_id INT NOT NULL,
            file_id VARCHAR(255) UNIQUE NOT NULL,
            project_name VARCHAR(255) NOT NULL,
            upload_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            status VARCHAR(50) NOT NULL DEFAULT 'Pending', -- e.g., Pending, Analyzing, Completed
            row_count INT,
            file_size INT,
            missing_values_count INT DEFAULT 0, 
            outlier_count INT DEFAULT 0         
        );
        """,
    ]

    with get_connection() as conn:
        with conn.cursor() as cursor:
            for query in schema_queries:
                cursor.execute(query)
            # Ensure the composite unique key exists for user and provider settings.
            cursor.execute(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'user_llm_settings_user_provider_key'
                    ) THEN
                        ALTER TABLE user_llm_settings ADD CONSTRAINT user_llm_settings_user_provider_key UNIQUE (user_id, provider);
                    END IF;
                END;
                $$;
            """
            )


# ==== User Management ====
def get_user_by_username(
    username: str,
    cursor: RealDictCursor,
) -> Optional[Dict[str, Any]]:
    """Fetches a user by their username from the database."""
    query = "SELECT id, username, hashed_password FROM users WHERE username = %s;"
    cursor.execute(query, (username,))
    user = cursor.fetchone()
    return user


# ==== Cleaning Logs ====
def save_cleaning_log(user_id: int, file_id: str, log: Dict[str, Any]) -> None:
    # This function is unchanged
    pass


# ==== Cleaning Sessions ====
def create_session(user_id: int, file_id: str, session_data: Dict[str, Any]) -> None:
    """
    Save or update a cleaning session in the PostgreSQL database.
    """
    query = """
        INSERT INTO cleaning_sessions (user_id, file_id, session_data, last_updated)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (user_id, file_id)
        DO UPDATE SET session_data = EXCLUDED.session_data,
                      last_updated = EXCLUDED.last_updated;
    """
    session_data_json = json.dumps(session_data)
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (user_id, file_id, session_data_json, datetime.now()))


def get_session(user_id: int, file_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a cleaning session from the PostgreSQL database."""
    query = "SELECT session_data FROM cleaning_sessions WHERE user_id = %s AND file_id = %s;"
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (user_id, file_id))
            result = cursor.fetchone()
            if result:
                return result[0]
            return None


def delete_session(user_id: int, file_id: str) -> None:
    """Delete a cleaning session from the database."""
    query = "DELETE FROM cleaning_sessions WHERE user_id = %s AND file_id = %s;"
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (user_id, file_id))


# ==== Project Tracking ====
def create_project_entry(project_data: Dict[str, Any], cursor) -> None:
    """
    Creates a new project record in the projects table.
    """
    query = """
        INSERT INTO projects
            (user_id, file_id, project_name, row_count, file_size, missing_values_count, outlier_count, status)
        VALUES
            (%(user_id)s, %(file_id)s, %(project_name)s, %(row_count)s, %(file_size)s, %(missing_values_count)s, %(outlier_count)s, 'Pending');
    """
    cursor.execute(query, project_data)


# ==== User LLM Settings ====
def save_llm_settings(user_id: int, settings: dict, cursor):
    """
    Saves or updates a user's LLM settings for a specific provider.
    """
    provider = settings.get("provider")
    api_key = settings.get("api_key")
    encrypted_api_key = encrypt(api_key)
    model_id = settings.get("model_id")
    endpoint_url = settings.get("endpoint_url")

    query = """
        INSERT INTO user_llm_settings (user_id, provider, api_key, model_id, endpoint_url, updated_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        ON CONFLICT (user_id, provider)
        DO UPDATE SET 
            api_key = EXCLUDED.api_key,
            model_id = EXCLUDED.model_id,
            endpoint_url = EXCLUDED.endpoint_url,
            updated_at = NOW();
    """
    cursor.execute(
        query, (user_id, provider, encrypted_api_key, model_id, endpoint_url)
    )


def get_all_llm_settings_for_user(user_id: int) -> List[Dict[str, Any]]:
    """Retrieve all saved LLM settings for a given user."""
    query = "SELECT provider, model_id FROM user_llm_settings WHERE user_id = %s;"
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (user_id,))
            return cursor.fetchall()


# ==== Retrieve Active LLM Setting ====
def get_active_llm_settings_for_user(
    user_id: int, cursor: RealDictCursor
) -> Optional[Dict[str, Any]]:
    """
    Retrieves and decrypts the active LLM setting for a user.
    """
    try:
        query = """
            SELECT 
                s.provider, 
                s.api_key,
                s.model_id,
                COALESCE(s.endpoint_url, p.default_endpoint_url) AS final_endpoint_url
            FROM 
                user_llm_settings s
            LEFT JOIN 
                provider_endpoint p ON s.provider = p.identifier
            WHERE 
                s.user_id = %s
            ORDER BY 
                s.updated_at DESC
            LIMIT 1;
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

    except Exception as e:
        print(f"Database error retrieving active settings for user {user_id}: {e}")
        return None


def update_project_status(user_id: int, file_id: str, new_status: str, cursor) -> None:
    """Updates the status of a specific project."""
    query = "UPDATE projects SET status = %s WHERE user_id = %s AND file_id = %s;"
    cursor.execute(query, (new_status, user_id, file_id))


# ==== Dashboard Data Retrieval ====
def get_projects_for_user(user_id: int, cursor) -> List[Dict[str, Any]]:
    """Fetches all projects for a specific user, ordered by most recent."""
    query = """
        SELECT 
            id,
            file_id, 
            project_name, 
            status, 
            row_count, 
            upload_time 
        FROM 
            projects 
        WHERE 
            user_id = %s 
        ORDER BY 
            upload_time DESC;
    """
    cursor.execute(query, (user_id,))
    return cursor.fetchall()


# === Dashboard Insights ====
def get_dashboard_insights(user_id: int, cursor) -> Dict[str, Any]:
    """Calculates and returns key insights for the user's dashboard."""

    # 1. Get count of datasets with missing values uploaded in the last 7 days
    query_missing = """
        SELECT COUNT(*) FROM projects
        WHERE user_id = %s AND missing_values_count > 0
        AND upload_time >= NOW() - INTERVAL '7 days';
    """
    cursor.execute(query_missing, (user_id,))
    missing_values_count = cursor.fetchone()["count"]

    # 2. Get count of datasets with outliers (using placeholder count for now)
    query_outliers = """
        SELECT COUNT(*) FROM projects
        WHERE user_id = %s AND outlier_count > 0
        AND upload_time >= NOW() - INTERVAL '7 days';
    """
    cursor.execute(query_outliers, (user_id,))
    outliers_detected_count = cursor.fetchone()["count"]

    # 3. Get average dataset size (row count)
    query_avg_size = (
        "SELECT AVG(row_count) as avg_size FROM projects WHERE user_id = %s;"
    )
    cursor.execute(query_avg_size, (user_id,))
    # Handle case where there are no projects yet
    avg_result = cursor.fetchone()["avg_size"]
    average_dataset_size = int(avg_result) if avg_result is not None else 0

    return {
        "missing_values_found_in_datasets": missing_values_count,
        "outliers_detected_in_datasets": outliers_detected_count,
        "average_dataset_size_in_rows": average_dataset_size,
    }
