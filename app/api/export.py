# app/api/export.py
import io
import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from app.api.clean import get_or_create_cleaner
from app.services.db_queries import update_project_status
from app.services.postgres_client import get_connection
from app.services.security import get_current_user

router = APIRouter(prefix="/export", tags=["export"])

# --- Pydantic Models ---


# UPDATED: Added user_id to all relevant request models
class ExportRequest(BaseModel):
    file_id: str
    # user_id: int
    format: str
    include_report: bool = False
    filename: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    columns: Optional[List[str]] = None


class MultiFormatExportRequest(BaseModel):
    file_id: str
    # user_id: int
    formats: List[str]
    include_report: bool = True
    include_original: bool = False


# --- Helper Functions ---


def generate_filename(
    file_id: str, format_str: str, custom_name: Optional[str] = None
) -> str:
    """Generate appropriate filename for export"""
    if custom_name:
        base_name = custom_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"cleaned_data_{file_id[:8]}_{timestamp}"

    return f"{base_name}.{format_str}"


def _format_bytes(num_bytes: float) -> str:
    """Format bytes to a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


# --- Endpoints ---


@router.post("/download")
async def export_data(
    request: ExportRequest, current_user: dict = Depends(get_current_user)
):
    """
    Export cleaned data based on the current frontend state and mark the project as completed.
    It prioritizes data sent in the request body, falling back to server-side session data if not provided.
    """
    user_id = current_user["user_id"]
    try:
        # Get the cleaner session to access the cleaning log, which might be needed for reports.
        cleaner = get_or_create_cleaner(user_id, request.file_id)
        df = None

        # --- MODIFIED LOGIC START ---
        # Prioritize data sent from the frontend if it exists
        if request.data is not None and request.columns is not None:
            print("--- Creating export from data payload received from frontend ---")
            df = pd.DataFrame(request.data, columns=request.columns)

            # Remove the internal '__id' column if it's present
            if "__id" in df.columns:
                df = df.drop(columns=["__id"])
        else:
            # Fallback to the server-side session data if no data is in the request
            print("--- Creating export from saved server-side session data ---")
            df = cleaner.df
        # --- MODIFIED LOGIC END ---

        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No data to export")

        filename = generate_filename(request.file_id, request.format, request.filename)

        if request.format.lower() == "csv":
            output = io.StringIO()
            df.to_csv(output, index=False)
            content = output.getvalue().encode("utf-8")
            media_type = "text/csv"

        elif request.format.lower() == "xlsx":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Cleaned_Data", index=False)
                if request.include_report and cleaner.cleaning_log:
                    report_df = pd.DataFrame(cleaner.cleaning_log)
                    report_df.to_excel(writer, sheet_name="Cleaning_Log", index=False)
            content = output.getvalue()
            media_type = (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        elif request.format.lower() == "json":
            content = df.to_json(orient="records", indent=2).encode("utf-8")
            media_type = "application/json"

        elif request.format.lower() == "parquet":
            output = io.BytesIO()
            df.to_parquet(output, index=False)
            content = output.getvalue()

        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported format: {request.format}"
            )

        # Update project status to 'Completed' after successful file generation
        with get_connection() as conn:
            with conn.cursor() as cursor:
                update_project_status(user_id, request.file_id, "Completed", cursor)

        # Finally, return the file to the user
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        # Log the full exception for better debugging
        print(f"An unexpected error occurred during export: {e}")
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")


@router.post("/multiple")
async def export_multiple_formats(
    request: MultiFormatExportRequest, current_user: dict = Depends(get_current_user)
):
    """Export data in multiple formats as a ZIP file."""
    user_id = current_user["user_id"]
    try:
        cleaner = get_or_create_cleaner(user_id, request.file_id)
        df = cleaner.df

        if df.empty:
            raise HTTPException(status_code=400, detail="No data to export")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for format_type in set(request.formats):  # Use set to avoid duplicates
                filename = generate_filename(request.file_id, format_type)

                if format_type.lower() == "csv":
                    zip_file.writestr(filename, df.to_csv(index=False))
                elif format_type.lower() == "xlsx":
                    output = io.BytesIO()
                    df.to_excel(output, index=False, sheet_name="Cleaned_Data")
                    zip_file.writestr(filename, output.getvalue())
                elif format_type.lower() == "json":
                    zip_file.writestr(filename, df.to_json(orient="records", indent=2))
                elif format_type.lower() == "parquet":
                    output = io.BytesIO()
                    df.to_parquet(output, index=False)
                    zip_file.writestr(filename, output.getvalue())

            if request.include_report and cleaner.cleaning_log:
                report_filename = generate_filename(
                    request.file_id, "json", "cleaning_report"
                )
                report_content = json.dumps(cleaner.cleaning_log, indent=2, default=str)
                zip_file.writestr(report_filename, report_content)

        zip_buffer.seek(0)
        zip_filename = f"data_export_{request.file_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creating multi-format export: {str(e)}"
        )


# UPDATED: Route now includes user_id
@router.get("/report/{file_id}")
async def get_cleaning_report(
    file_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get the cleaning report as a JSON response."""
    user_id = current_user["user_id"]
    try:
        cleaner = get_or_create_cleaner(user_id, file_id)
        if not cleaner.cleaning_log:
            raise HTTPException(
                status_code=404,
                detail="No cleaning actions have been performed for this session.",
            )

        report = {"summary": cleaner.get_data_summary(), "log": cleaner.cleaning_log}
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching report: {str(e)}")


@router.get("/formats")
async def get_supported_formats():
    """Get a list of supported export formats."""
    return {
        "csv": "Comma-Separated Values",
        "xlsx": "Microsoft Excel",
        "json": "JSON",
        "parquet": "Apache Parquet",
    }


# UPDATED: Route now includes user_id
@router.get("/status/{file_id}")
async def get_export_status(
    file_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get export status and available options for a session."""
    user_id = current_user["user_id"]
    try:
        cleaner = get_or_create_cleaner(user_id, file_id)
        df = cleaner.df

        return {
            "file_id": file_id,
            "data_available": not df.empty,
            "has_cleaning_session": bool(cleaner.cleaning_log),
            "export_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
                "memory_usage_human": _format_bytes(df.memory_usage(deep=True).sum()),
            },
            "available_formats": ["csv", "xlsx", "json", "parquet"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting export status: {str(e)}"
        )
