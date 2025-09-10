# app/routers/ai_settings.py

import logging
from typing import Any, Dict, List, Optional

import httpx
import psycopg2
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor

from app.services.db_queries import (
    get_active_llm_settings_for_user,
    get_all_llm_settings_for_user,
    save_llm_settings,
)
from app.services.postgres_client import get_connection, put_connection
from app.services.security import get_current_user

# Router setup
router = APIRouter(prefix="/ai", tags=["ai"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Database Dependency ---
def get_db_cursor():
    """Provides a database cursor that automatically closes the connection."""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            yield cursor
        conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed.")
    finally:
        if conn:
            put_connection(conn)


# --- Pydantic Models ---
class SetActiveProviderRequest(BaseModel):
    # user_id: int
    provider: str


class LLMSettings(BaseModel):
    provider: str
    model_id: Optional[str] = None
    api_key: Optional[str] = None
    self_hosted_endpoint: Optional[str] = None  # Corrected from endpoint_url


class SaveSettingsRequest(BaseModel):
    # user_id: int
    settings: LLMSettings


class GetSettingsRequest(BaseModel):
    # user_id: int
    pass


class AIModel(BaseModel):
    model_id: str
    name: str


class AIProvider(BaseModel):
    identifier: str
    name: str


# --- Endpoint for Frontend Provider Dropdown ---
@router.get("/settings/providers")
async def get_user_providers_for_dropdown(
    cursor: RealDictCursor = Depends(get_db_cursor),
    current_user: dict = Depends(get_current_user),
):
    """
    Retrieves all of a user's saved providers and identifies which one is
    currently active. This is used to populate the frontend dropdown.
    """
    user_id = current_user["id"]
    try:
        # First, find the single active provider to identify it in the response
        active_settings = get_active_llm_settings_for_user(user_id, cursor)
        active_provider_id = active_settings["provider"] if active_settings else None

        # Then, get the list of all providers the user has configured
        all_providers = get_all_llm_settings_for_user(user_id, cursor)

        return {
            "status": "success",
            "data": {"providers": all_providers, "active_provider": active_provider_id},
        }
    except Exception as e:
        logger.error(f"Database error retrieving providers for user {user_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve provider settings."
        )


# --- Helper function for parsing API responses ---
def parse_models_response(
    provider_id: str, data: Dict[str, Any]
) -> List[Dict[str, str]]:
    try:
        if provider_id in [
            "openai",
            "deepseek",
            "anthropic",
            "fireworksai",
            "mistral",
            "openrouter",
        ]:
            models_list = data.get("data", [])
            if not isinstance(models_list, list):
                return []
            if provider_id == "openai":
                models_list = sorted(
                    [
                        m
                        for m in models_list
                        if isinstance(m, dict) and "id" in m and "gpt" in m["id"]
                    ],
                    key=lambda x: x.get("id"),
                    reverse=True,
                )
            return [
                {"model_id": m["id"], "name": m["id"]}
                for m in models_list
                if isinstance(m, dict) and "id" in m
            ]
        elif provider_id == "cohere":
            models_list = data.get("models", [])
            if not isinstance(models_list, list):
                return []
            chat_models = [m for m in models_list if "chat" in m.get("endpoints", [])]
            return [
                {"model_id": m["name"], "name": m["name"]}
                for m in chat_models
                if isinstance(m, dict) and "name" in m
            ]
        elif provider_id == "ollama":
            models_list = data.get("models", [])
            if not isinstance(models_list, list):
                return []
            return [
                {"model_id": m["name"], "name": m["name"]}
                for m in models_list
                if isinstance(m, dict) and "name" in m
            ]
        elif provider_id == "google":
            models_list = data.get("models", [])
            if not isinstance(models_list, list):
                return []
            return [
                {
                    "model_id": (m.get("name") or "unknown").split("/")[-1],
                    "name": m.get("displayName") or m.get("name") or "unknown",
                }
                for m in models_list
                if isinstance(m, dict) and "name" in m
            ]
        return []
    except Exception as e:
        logger.error(f"Failed to parse models for {provider_id}: {e}")
        return []


# --- Dynamic Provider & Model Endpoints ---
@router.get("/providers/{provider_id}/models")
async def list_models_for_provider(
    provider_id: str,
    api_key: str = Header(..., description="The API key for the selected provider."),
):
    provider_api_map = {
        "openai": "https://api.openai.com/v1/models",
        "deepseek": "https://api.deepseek.com/v1/models",
        "cohere": "https://api.cohere.com/v1/models",
        "anthropic": "https://api.anthropic.com/v1/models",
        "fireworksai": "https://api.fireworks.ai/inference/v1/models",
        "openrouter": "https://openrouter.ai/api/v1/models",
        "mistral": "https://api.mistral.ai/v1/models",
        "ollama": "http://localhost:11434/api/tags",
        "google": "https://generativelanguage.googleapis.com/v1beta/models",
    }
    url = provider_api_map.get(provider_id)
    if not url:
        return JSONResponse(content=[], status_code=200)

    headers = (
        {"x-goog-api-key": api_key}
        if provider_id == "google"
        else {"Authorization": f"Bearer {api_key}"}
    )
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            raw = resp.json() or {}
        models = parse_models_response(provider_id, raw)
        safe_models = [
            {"model_id": m.get("model_id"), "name": m.get("name")}
            for m in models
            if isinstance(m, dict)
        ]
        return JSONResponse(content=safe_models, status_code=200)
    except Exception as e:
        logger.error(f"Error fetching models for {provider_id}: {e}")
        return JSONResponse(content=[], status_code=200)


# --- NEW: Endpoint to set the active provider ---
@router.post("/settings/active")
async def set_active_provider(
    req: SetActiveProviderRequest,
    cursor: RealDictCursor = Depends(get_db_cursor),
    current_user: dict = Depends(get_current_user),
):
    """
    Sets a provider as active by updating its 'updated_at' timestamp.
    """
    user_id = current_user["id"]
    try:
        # This query finds the user's specific setting for a provider
        # and updates its timestamp to the current time.
        query = """
            UPDATE user_llm_settings
            SET updated_at = NOW()
            WHERE user_id = %s AND provider = %s;
        """
        cursor.execute(query, (user_id, req.provider))

        if cursor.rowcount == 0:
            logger.warning(
                f"Attempted to set active provider for a non-existent setting: User {user_id}, Provider {req.provider}"
            )

        return {
            "status": "success",
            "message": f"Provider '{req.provider}' is now active.",
        }
    except psycopg2.Error as e:
        logger.error(f"Database error setting active provider: {e}")
        raise HTTPException(status_code=500, detail="Database error updating setting.")


# --- User Settings Endpoints ---
@router.post("/settings")
async def save_user_llm_settings(
    req: SaveSettingsRequest,
    cursor: RealDictCursor = Depends(get_db_cursor),
    current_user: dict = Depends(get_current_user),
):
    """Saves or updates a user's LLM settings by calling the database client."""
    user_id = current_user["id"]
    try:
        settings = req.settings.model_dump()
        save_llm_settings(user_id=user_id, settings=settings, cursor=cursor)
        return {"status": "success", "message": "Settings saved successfully"}
    except psycopg2.Error as e:
        logger.error(f"Database error saving settings: {e}")
        raise HTTPException(
            status_code=500, detail=f"Database error saving settings: {e}"
        )


@router.post("/settings/get")
async def get_user_llm_settings(
    req: GetSettingsRequest,
    cursor: RealDictCursor = Depends(get_db_cursor),
    current_user: dict = Depends(get_current_user),
):
    """Retrieves a user's LLM settings from the database."""
    user_id = current_user["id"]
    try:
        query = """
            SELECT provider, api_key, model_id, endpoint_url AS self_hosted_endpoint
            FROM user_llm_settings
            WHERE user_id = %s;
        """
        cursor.execute(query, (user_id,))
        settings = cursor.fetchone()
        if not settings:
            raise HTTPException(
                status_code=404, detail="No settings found for this user."
            )
        return {"status": "success", "data": settings}
    except HTTPException:
        raise
    except psycopg2.Error as e:
        logger.error(f"Database error retrieving settings: {e}")
        raise HTTPException(
            status_code=500, detail=f"Database error retrieving settings: {e}"
        )


@router.post("/settings/all")
async def get_all_user_llm_settings(
    req: GetSettingsRequest,
    cursor: RealDictCursor = Depends(get_db_cursor),
    current_user: dict = Depends(get_current_user),
):
    """Retrieves all saved LLM settings for a given user from the database."""
    user_id = current_user["id"]
    try:
        query = """
            SELECT provider, api_key, model_id, endpoint_url AS self_hosted_endpoint
            FROM user_llm_settings
            WHERE user_id = %s;
        """
        cursor.execute(query, (user_id,))
        settings_list = cursor.fetchall()
        return {"status": "success", "data": settings_list}
    except psycopg2.Error as e:
        logger.error(f"Database error retrieving all settings: {e}")
        raise HTTPException(
            status_code=500, detail=f"Database error retrieving all settings: {e}"
        )


# --- Connection Test Endpoint ---
@router.post("/test-connection")
async def test_llm_connection(settings: LLMSettings):
    if not settings.api_key and settings.provider != "self-hosted":
        raise HTTPException(
            status_code=400, detail="API key is required for this provider."
        )

    provider_endpoints = {
        "openai": "https://api.openai.com/v1/models",
        "deepseek": "https://api.deepseek.com/v1/models",
        "cohere": "https://api.cohere.com/v1/models",
        "anthropic": "https://api.anthropic.com/v1/models",
        "fireworksai": "https://api.fireworks.ai/inference/v1/models",
        "openrouter": "https://openrouter.ai/api/v1/models",
        "mistral": "https://api.mistral.ai/v1/models",
        "google": "https://generativelanguage.googleapis.com/v1beta/models",
    }
    url = provider_endpoints.get(settings.provider)
    if not url:
        return {"status": "success", "message": "Configuration is valid."}

    if settings.provider == "google":
        headers = {"x-goog-api-key": settings.api_key}
    else:
        headers = {"Authorization": f"Bearer {settings.api_key}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        return {"status": "success", "message": "Connection successful!"}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="Authentication failed. The API key is invalid or expired.",
            )
        else:
            logger.error(
                f"API Error during connection test: {e.response.status_code} - {e.response.text}"
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"API Error: {e.response.text}",
            )
    except httpx.RequestError as e:
        logger.error(f"Network error during connection test: {e}")
        raise HTTPException(
            status_code=500, detail="Network error: Could not connect to the service."
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during connection test: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )
