# app/schemas/report.py

from pydantic import BaseModel
from typing import List, Dict, Any

class ReportRequest(BaseModel):
    """
    Pydantic model for the incoming report generation request.
    """
    current_data: List[Dict[str, Any]]
    project_name: str