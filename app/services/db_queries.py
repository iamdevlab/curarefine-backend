# app/services/db_queries.py

import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from psycopg2.extras import RealDictCursor

from app.services.encryption_service import encrypt, decrypt
from app.services.postgres_client import get_connection, put_connection


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


# def create_project_entry(project_data: Dict[str, Any], cursor: RealDictCursor):
#     query = """
#         INSERT INTO projects
#             (user_id, file_id, project_name, row_count, file_size, missing_values_count, outlier_count, status)
#         VALUES
#             (%(user_id)s, %(file_id)s, %(project_name)s, %(row_count)s, %(file_size)s, %(missing_values_count)s, %(outlier_count)s, 'Pending');
#         """
#     cursor.execute(query, project_data)


def update_project_status(
    user_id: int, file_id: str, new_status: str, cursor: RealDictCursor
):
    query = "UPDATE projects SET status = %s WHERE user_id = %s AND file_id = %s;"
    cursor.execute(query, (new_status, user_id, file_id))


def save_llm_settings(user_id: int, settings: dict, cursor: RealDictCursor):
    provider = settings.get("provider")
    encrypted_api_key = encrypt(settings.get("api_key"))
    model_id = settings.get("model_id")
    endpoint_url = settings.get("self_hosted_endpoint")
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


def get_llm_settings_for_provider(
        user_id: int, provider: str, cursor: RealDictCursor
) -> Optional[Dict[str, Any]]:
    """
    Fetches user-specific settings (API key, model) and combines them
    with the centrally-managed endpoint URL for the given provider.
    """
    # Step 1: Get the user's specific settings (API key, model_id, etc.)
    user_settings_query = """
                          SELECT provider, api_key, model_id, endpoint_url AS self_hosted_endpoint
                          FROM user_llm_settings
                          WHERE user_id = %s \
                            AND provider = %s; \
                          """
    cursor.execute(user_settings_query, (user_id, provider))
    settings = cursor.fetchone()

    if not settings:
        return None  # User has not configured this provider

    settings = dict(settings)
    if settings.get("api_key"):
        settings["api_key"] = decrypt(settings["api_key"])

    # Step 2: If the provider is not self-hosted, get the URL from the admin table.
    if provider != "self-hosted":
        admin_endpoint_query = """
         SELECT default_endpoint_url FROM provider_endpoint WHERE identifier = %s;
        """
        cursor.execute(admin_endpoint_query, (provider,))
        endpoint_data = cursor.fetchone()

        # Replace the (likely null) self_hosted_endpoint with the official one
        if endpoint_data:
            settings["self_hosted_endpoint"] = endpoint_data['url']
        else:
            # Handle case where provider is not in the admin table
            settings["self_hosted_endpoint"] = None

    return settings
# def get_llm_settings_for_provider(
#     user_id: int, provider: str, cursor: RealDictCursor
# ) -> Optional[Dict[str, Any]]:
#     """Fetches the settings for a specific provider for a user."""
#     query = """
#         SELECT provider, api_key, model_id, endpoint_url AS self_hosted_endpoint
#         FROM user_llm_settings
#         WHERE user_id = %s AND provider = %s;
#     """
#     cursor.execute(query, (user_id, provider))
#     settings = cursor.fetchone()
#     if settings:
#         settings = dict(settings)
#         if settings.get("api_key"):
#             settings["api_key"] = decrypt(settings["api_key"])
#         return settings
#     return None

def get_active_llm_settings_for_user(
    user_id: int, cursor: RealDictCursor
) -> Optional[Dict[str, Any]]:
    query = """
        SELECT provider, api_key, model_id, endpoint_url
        FROM user_llm_settings
        WHERE user_id = %s ORDER BY updated_at DESC LIMIT 1;
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
        SELECT id, file_id, file_url, project_name, status, row_count, upload_time 
        FROM projects 
        WHERE user_id = %s 
        ORDER BY upload_time DESC;
    """
    cursor.execute(query, (user_id,))
    return cursor.fetchall()


# def get_projects_for_user(user_id: int, cursor: RealDictCursor) -> List[Dict[str, Any]]:
#     query = """
#         SELECT id, file_id, project_name, status, row_count, upload_time
#         FROM projects WHERE user_id = %s ORDER BY upload_time DESC;
#     """
#     cursor.execute(query, (user_id,))
#     return cursor.fetchall()


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
