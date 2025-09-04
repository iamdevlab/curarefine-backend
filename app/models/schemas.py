from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from enum import Enum


class FileType(str, Enum):
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"
    WORD = "word"
    TEXT = "text"


class UploadResponse(BaseModel):
    filename: str
    file_type: FileType
    size: int
    rows: int
    columns: int
    column_names: List[str]
    data_types: Dict[str, str]
    preview: List[Dict[str, Any]]
    issues: List[Dict[str, Any]]


class CleaningRule(BaseModel):
    column: str
    rule_type: str
    action: str
    confidence: float


class CleaningRequest(BaseModel):
    file_id: str
    rules: List[CleaningRule]
    auto_apply: bool = False


class ExportRequest(BaseModel):
    file_id: str
    format: str
    include_metadata: bool = True
