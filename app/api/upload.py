from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from app.utils.file_parser import read_file_content
import pandas as pd
import io
import uuid
from typing import List, Optional
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import math
from app.services.postgres_client import get_connection, create_project_entry
from app.services.security import get_current_user


router = APIRouter(tags=["upload"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
PREVIEW_ROW_LIMIT = 100  # Maximum rows to return in preview


def validate_file(file: UploadFile) -> bool:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    return True


def read_file_content(content: bytes, filename: str) -> pd.DataFrame:
    try:
        file_ext = Path(filename).suffix.lower()

        if file_ext == ".csv":
            # Try reading with headers (assume first row is header)
            df = pd.read_csv(io.BytesIO(content), header=0)
            # If pandas auto-generated generic headers (e.g. "Unnamed: 0"),
            # replace them with column_1, column_2, ...
            if all(str(col).startswith("Unnamed") for col in df.columns):
                df = pd.read_csv(io.BytesIO(content), header=None)
                df.columns = [f"column_{i+1}" for i in range(len(df.columns))]

        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(io.BytesIO(content), header=0)
            if all(str(col).startswith("Unnamed") for col in df.columns):
                df = pd.read_excel(io.BytesIO(content), header=None)
                df.columns = [f"column_{i+1}" for i in range(len(df.columns))]

        elif file_ext == ".json":
            df = pd.read_json(io.BytesIO(content))

            # Ensure columns are named properly if JSON is array of arrays
            if isinstance(df.columns, pd.RangeIndex):
                df.columns = [f"column_{i+1}" for i in range(len(df.columns))]
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        return df

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")


@router.post("/file")
async def upload_file(
    file: UploadFile = File(...), current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    validate_file(file)

    content = await file.read()
    file_size = len(content)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    # This correctly uses your separate utility function from app/utils/file_parser.py
    df = read_file_content(content, file.filename)

    try:
        df = df.replace({np.nan: None})
        total_missing = int(df.isnull().sum().sum())
        # (Placeholder for future outlier detection logic)
        total_outliers = 0  # In the future, you can add a real calculation here.

        file_id = str(uuid.uuid4())
        _, ext = os.path.splitext(file.filename)
        ext = ext.lower()

        if not ext:
            raise HTTPException(status_code=400, detail="File must have an extension")

        stored_filename = f"{file_id}{ext}"
        file_path = UPLOAD_DIR / stored_filename

        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # --- NEW: Create a project entry in the database ---
        project_data = {
            "user_id": user_id,
            "file_id": stored_filename,
            "project_name": file.filename,
            "row_count": len(df),
            "file_size": file_size,
            "missing_values_count": total_missing,
            "outlier_count": total_outliers,
        }
        with get_connection() as conn:
            with conn.cursor() as cursor:
                create_project_entry(project_data, cursor)
        # --- END NEW ---

        preview_size = min(PREVIEW_ROW_LIMIT, len(df))

        file_info = {
            "file_id": stored_filename,
            "original_filename": file.filename,
            "size": file_size,
            "upload_time": datetime.now().isoformat(),
            "row_count": len(df),
            "columns": df.columns.tolist(),
            "data_types": {col: str(df[col].dtype) for col in df.columns},
            "file_path": str(file_path),
            "preview_size": preview_size,
        }

        preview_data = df.head(preview_size).to_dict("records")

        quality_summary = {
            "missing_values": df.isnull().sum().to_dict(),
            "total_missing": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage": int(df.memory_usage(deep=True).sum()),
        }

        return JSONResponse(
            {
                "status": "success",
                "message": "File uploaded and project created successfully",
                "file_info": file_info,
                "preview": preview_data,
                "quality_summary": quality_summary,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/preview/{file_id}")
async def preview_file(file_id: str, page: int = 1, limit: int = 50):
    try:
        # Build full path (file_id already includes extension)
        file_path = UPLOAD_DIR / file_id

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Parse file using helper (handles csv, excel, json, txt, etc.)
        df = read_file_content(file_path.read_bytes(), file_path.name)
        # Replace numpy's Not-a-Number with Python's None for JSON compatibility
        df = df.replace({np.nan: None})
        total_rows = len(df)
        start = (page - 1) * limit
        end = start + limit

        # Slice preview data
        preview_data = df.iloc[start:end].to_dict(orient="records")

        response = {
            "file_id": file_id,
            "filename": file_path.name,
            "columns": list(df.columns),
            "data": preview_data,
            "total_rows": total_rows,
            "preview_size": len(preview_data),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "quality_summary": {
                "missing_values": df.isnull().sum().to_dict(),
                "total_missing": int(df.isnull().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
                "memory_usage": int(df.memory_usage(deep=True).sum()),
            },
            "page": page,
            "limit": limit,
            "total_pages": (total_rows + limit - 1) // limit,
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error previewing file: {str(e)}")


# Keep the rest of your existing endpoints (multiple uploads, file info, etc.)
