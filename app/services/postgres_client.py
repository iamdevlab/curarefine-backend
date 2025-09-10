# app/services/postgres_client.py

import os
from fastapi import HTTPException
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

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
                minconn=1, maxconn=20, dsn=db_url
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


# --- Schema Initialization ---
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
