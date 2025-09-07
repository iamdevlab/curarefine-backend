# app/routers/cleaning.py
import io
import os
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.services.data_cleaner import DataCleaner
from app.services.db_queries import (
    create_session,
    delete_session as pg_delete_session,
    get_session as pg_get_session,
    update_project_status,
)
from app.services.postgres_client import get_connection
from app.services.redis_client import (
    delete_session as redis_delete_session,
    get_session as redis_get_session,
    set_session,
)
from app.services.security import get_current_user
from google.cloud import storage


router = APIRouter(prefix="/clean", tags=["cleaning"])
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")
UPLOAD_DIR = Path("uploads")


# --- Pydantic Models ---
class MissingValuesRequest(BaseModel):
    file_id: str
    strategy: str
    columns: Optional[List[str]] = None
    fill_value: Optional[Any] = None
    # user_id: int


class DuplicatesRequest(BaseModel):
    file_id: str
    subset: Optional[List[str]] = None
    keep: str = "first"
    # user_id: int


class TextCleaningRequest(BaseModel):
    file_id: str
    columns: Optional[List[str]] = None
    operations: List[str] = ["strip", "lower", "remove_special_chars"]
    # user_id: int


class OutliersRequest(BaseModel):
    file_id: str
    columns: Optional[List[str]] = None
    method: str = "iqr"
    action: str = "remove"
    # user_id: int


class ConvertTypesRequest(BaseModel):
    file_id: str
    conversions: Dict[str, str]
    # user_id: int


class StandardizeColumnsRequest(BaseModel):
    file_id: str
    # user_id: int


class ProjectState(BaseModel):
    rows: List[Dict[str, Any]]
    columns: List[str]
    undoStack: List[Any] = []
    redoStack: List[Any] = []
    actionHistory: List[Any] = []
    originalRows: List[Dict[str, Any]] = []
    originalColumns: List[str] = []


class SaveProjectRequest(BaseModel):
    file_id: str
    # user_id: int
    state: ProjectState


def get_dataframe(file_id: str) -> pd.DataFrame:
    """
    Loads a DataFrame either from local uploads/ (dev) or from GCS (prod).
    """
    try:
        # --- 1. Try GCS first if bucket is set ---
        if GCS_BUCKET_NAME:
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(file_id)

            if not blob.exists():
                raise HTTPException(status_code=404, detail="File not found in GCS")

            # download into memory
            content = blob.download_as_bytes()
            suffix = Path(file_id).suffix.lower()

            if suffix == ".csv":
                return pd.read_csv(io.BytesIO(content))
            elif suffix in [".xlsx", ".xls"]:
                return pd.read_excel(io.BytesIO(content))
            elif suffix == ".json":
                return pd.read_json(io.BytesIO(content))
            elif suffix == ".parquet":
                return pd.read_parquet(io.BytesIO(content))
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

        # --- 2. Fallback to local (dev mode) ---
        file_path = UPLOAD_DIR / file_id
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(file_path)
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        elif suffix == ".json":
            return pd.read_json(file_path)
        elif suffix == ".parquet":
            return pd.read_parquet(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


# --- Helpers ---
# def get_dataframe(file_id: str) -> pd.DataFrame:
#     file_path = UPLOAD_DIR / file_id
#     if not file_path.exists():
#         raise HTTPException(status_code=404, detail="File not found")

#     try:
#         suffix = file_path.suffix.lower()
#         if suffix == ".csv":
#             return pd.read_csv(file_path)
#         elif suffix in [".xlsx", ".xls"]:
#             return pd.read_excel(file_path)
#         elif suffix == ".json":
#             return pd.read_json(file_path)
#         elif suffix == ".parquet":
#             return pd.read_parquet(file_path)
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported file type")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


def get_or_create_cleaner(user_id: int, file_id: str) -> DataCleaner:
    try:
        # user_id = current_user["user_id"]
        session_data = redis_get_session(user_id, file_id)

        if session_data:
            return DataCleaner(pd.DataFrame(session_data))

        df = get_dataframe(file_id)
        cleaner = DataCleaner(df)
        set_session(user_id, file_id, cleaner.df.to_dict())
        return cleaner

    except Exception as e:
        print("--- A CRITICAL ERROR OCCURRED IN get_or_create_cleaner ---")
        print(f"ERROR: {e}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        raise e


def prepare_df_for_serialization(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype.name in ["category", "datetime64[ns]"]:
            df_copy[col] = df_copy[col].astype(str)
    return df_copy


def build_cleaning_response(
    cleaner: DataCleaner,
    # user_id: int,
    file_id: str,
    message: str,
):
    try:
        log = cleaner.cleaning_log[-1]
        serializable_df = prepare_df_for_serialization(cleaner.df)
        set_session(user_id, file_id, serializable_df.to_dict())
        summary = cleaner.get_data_summary()

        return {
            "status": "success",
            "message": message,
            "updated_summary": summary,
            "cleaning_log": log,
        }
    except Exception as e:
        print("--- A CRITICAL ERROR OCCURRED INSIDE build_cleaning_response ---")
        print(f"ERROR: {e}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        raise e


# --- Endpoints ---
@router.post("/save")
async def save_project_state(
    request: SaveProjectRequest, current_user: dict = Depends(get_current_user)
):
    """
    Receives the full frontend state, saves it, and updates the project status.
    """
    user_id = current_user["user_id"]
    try:
        state_dict = request.state.model_dump()

        # Manage connection and cursor here to ensure both actions succeed or fail together
        with get_connection() as conn:
            with conn.cursor() as cursor:
                # 1. Save the actual session data
                create_session(user_id, request.file_id, state_dict)

                # 2. Update the project status to 'Analyzing'
                update_project_status(user_id, request.file_id, "Analyzing", cursor)

        return {
            "status": "success",
            "message": "Project state saved and status updated successfully.",
        }
    except TypeError as te:  # Specifically catch JSON serialization errors
        print(f"SERIALIZATION ERROR: {te}")
        raise HTTPException(
            status_code=400, detail=f"Data is not JSON serializable: {str(te)}"
        )
    except Exception as e:
        # Generic catch-all for other errors (e.g., database)
        print(f"SAVE FAILED: An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error saving project state: {str(e)}"
        )


@router.get("/load/{file_id}")
async def load_project_state(
    file_id: str, current_user: dict = Depends(get_current_user)
):
    """
    Checks for a saved session in the database. If it exists, returns the saved state.
    Otherwise, falls back to loading the original uploaded file.
    """
    user_id = current_user["user_id"]
    try:
        # Check the database for a saved session
        saved_state_data = pg_get_session(user_id, file_id)

        if saved_state_data:
            # --- FIX: Synchronize PostgreSQL state with Redis ---
            # This ensures that subsequent cleaning actions use the correct, saved data.
            if "rows" in saved_state_data:
                # The 'rows' key contains the main data grid content.
                # We save this part to Redis for the DataCleaner.
                set_session(user_id, file_id, saved_state_data["rows"])
            # A saved state was found in the database
            return {"savedState": saved_state_data}
        else:
            # No saved state, load the original file as a fallback
            df = get_dataframe(file_id)
            df = df.replace({np.nan: None})
            return {
                "data": df.to_dict(orient="records"),
                "columns": df.columns.tolist(),
            }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading project state: {str(e)}"
        )


@router.post("/missing")
async def handle_missing_values(
    request: MissingValuesRequest, current_user: dict = Depends(get_current_user)
):
    try:
        user_id = current_user["user_id"]
        cleaner = get_or_create_cleaner(user_id, request.file_id)
        cleaner.handle_missing_values(
            strategy=request.strategy,
            columns=request.columns,
            fill_value=request.fill_value,
        )
        return build_cleaning_response(
            cleaner,
            user_id,
            request.file_id,
            f"Missing values handled using strategy '{request.strategy}'",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error handling missing values: {str(e)}"
        )


@router.post("/duplicates")
async def remove_duplicates(
    request: DuplicatesRequest, current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    try:
        cleaner = get_or_create_cleaner(user_id, request.file_id)
        cleaner.remove_duplicates(subset=request.subset, keep=request.keep)
        return build_cleaning_response(
            cleaner,
            user_id,
            request.file_id,
            "Duplicates removed successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error removing duplicates: {str(e)}"
        )


@router.post("/text")
async def clean_text_columns(
    request: TextCleaningRequest, current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    try:
        cleaner = get_or_create_cleaner(user_id, request.file_id)
        # --- ADD LOGGING HERE ---
        if request.columns:
            col_to_log = request.columns[0]
            print(f"--- BEFORE CLEANING '{col_to_log}' ---")
            print(f"Data type: {cleaner.df[col_to_log].dtype}")
            print("Values:\n", cleaner.df[col_to_log].head())
            # --- END LOGGING ---
        cleaner.clean_text_columns(
            columns=request.columns, operations=request.operations
        )
        # --- ADD LOGGING HERE ---
        if request.columns:
            col_to_log = request.columns[0]
            print(f"--- AFTER CLEANING '{col_to_log}' ---")
            print(f"Data type: {cleaner.df[col_to_log].dtype}")
            print("Values:\n", cleaner.df[col_to_log].head())
            # --- END LOGGING ---

        return build_cleaning_response(
            cleaner,
            user_id,
            request.file_id,
            "Text columns cleaned successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error cleaning text columns: {str(e)}"
        )


@router.post("/outliers")
async def handle_outliers(
    request: OutliersRequest, current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    try:
        cleaner = get_or_create_cleaner(user_id, request.file_id)
        cleaner.handle_outliers(
            columns=request.columns,
            method=request.method,
            action=request.action,
        )
        return build_cleaning_response(
            cleaner,
            user_id,
            request.file_id,
            f"Outliers handled using method '{request.method}' and action '{request.action}'",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error handling outliers: {str(e)}"
        )


@router.post("/convert-types")
async def convert_data_types(
    request: ConvertTypesRequest, current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    try:
        cleaner = get_or_create_cleaner(user_id, request.file_id)
        cleaner.convert_data_types(conversions=request.conversions)
        return build_cleaning_response(
            cleaner,
            user_id,
            request.file_id,
            "Column data types converted successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error converting data types: {str(e)}"
        )


@router.post("/standardize-columns")
async def standardize_columns(
    request: StandardizeColumnsRequest, current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    try:
        cleaner = get_or_create_cleaner(user_id, request.file_id)
        cleaner.standardize_column_names()
        return build_cleaning_response(
            cleaner,
            user_id,
            request.file_id,
            "Column names standardized successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error standardizing columns: {str(e)}"
        )


@router.get("/preview/{file_id}")
async def preview_data(
    file_id: str,
    current_user: dict = Depends(get_current_user),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
):
    user_id = current_user["user_id"]
    try:
        cleaner = get_or_create_cleaner(user_id, file_id)
        serializable_df = prepare_df_for_serialization(cleaner.df)

        total_rows = len(serializable_df)
        start, end = (page - 1) * limit, (page - 1) * limit + limit
        rows = serializable_df.iloc[start:end].to_dict(orient="records")

        return {
            "rows": rows,
            "page": page,
            "limit": limit,
            "total_rows": total_rows,
            "total_pages": (total_rows // limit) + (1 if total_rows % limit else 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error previewing data: {str(e)}")


@router.get("/recommendations/{file_id}")
async def get_recommendations(
    file_id: str, current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    try:
        cleaner = get_or_create_cleaner(user_id, file_id)
        return {
            "status": "active",
            "updated_summary": cleaner.get_data_summary(),
            "cleaning_log": cleaner.cleaning_log,
            "next_recommendations": (
                cleaner.get_recommendations()
                if hasattr(cleaner, "get_recommendations")
                else []
            ),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting recommendations: {str(e)}"
        )


@router.delete("/session/{file_id}")
async def end_cleaning_session(
    file_id: str, current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    try:
        redis_delete_session(user_id, file_id)
        return {
            "status": "success",
            "message": "Cleaning session ended successfully",
            "file_id": file_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending session: {str(e)}")
