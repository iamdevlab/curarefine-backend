import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


def to_python_type(value: Any) -> Any:
    """Convert numpy/pandas scalars to Python-native types for JSON serialization."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta,)):
        return str(value)
    return value


class DataCleaner:
    """Core data cleaning functionality"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_df = df.copy()
        self.cleaning_log: List[Dict[str, Any]] = []

    # ========== LOGGING ==========
    def log_action(self, action: str, details: Dict[str, Any]):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": {k: to_python_type(v) for k, v in details.items()},
        }
        self.cleaning_log.append(log_entry)
        logger.info(f"Data cleaning action: {action} - {details}")

    # ========== SUMMARIES ==========
    def get_data_summary(self) -> Dict[str, Any]:
        return {
            "shape": tuple(map(int, self.df.shape)),
            "columns": self.df.columns.tolist(),
            "data_types": {k: str(v) for k, v in self.df.dtypes.to_dict().items()},
            "missing_values": {
                k: int(v) for k, v in self.df.isnull().sum().to_dict().items()
            },
            "duplicate_rows": int(self.df.duplicated().sum()),
            "memory_usage": int(self.df.memory_usage(deep=True).sum()),
            "numeric_columns": self.df.select_dtypes(
                include=[np.number]
            ).columns.tolist(),
            "text_columns": self.df.select_dtypes(include=["object"]).columns.tolist(),
            "datetime_columns": self.df.select_dtypes(
                include=["datetime64"]
            ).columns.tolist(),
        }

    def detect_issues(self) -> Dict[str, Any]:
        issues = {
            "missing_values": {},
            "duplicates": 0,
            "outliers": {},
            "whitespace_issues": {},
        }

        # Missing
        missing = self.df.isnull().sum()
        issues["missing_values"] = {
            col: int(count) for col, count in missing.items() if count > 0
        }

        # Duplicates
        issues["duplicates"] = int(self.df.duplicated().sum())

        # Outliers (IQR method)
        for col in self.df.select_dtypes(include=[np.number]).columns:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[
                (self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))
            ][col]
            if len(outliers) > 0:
                issues["outliers"][col] = int(len(outliers))

        # Whitespace
        for col in self.df.select_dtypes(include=["object"]).columns:
            leading = self.df[col].astype(str).str.startswith(" ").sum()
            trailing = self.df[col].astype(str).str.endswith(" ").sum()
            if leading > 0 or trailing > 0:
                issues["whitespace_issues"][col] = {
                    "leading": int(leading),
                    "trailing": int(trailing),
                }

        return issues

    def get_cleaning_report(self) -> Dict[str, Any]:
        original_summary = {
            "rows": int(self.original_df.shape[0]),
            "columns": int(self.original_df.shape[1]),
            "missing_values": int(self.original_df.isnull().sum().sum()),
            "duplicates": int(self.original_df.duplicated().sum()),
        }
        current_summary = {
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1]),
            "missing_values": int(self.df.isnull().sum().sum()),
            "duplicates": int(self.df.duplicated().sum()),
        }

        return {
            "original_data": original_summary,
            "cleaned_data": current_summary,
            "changes": {
                "rows_removed": original_summary["rows"] - current_summary["rows"],
                "columns_removed": original_summary["columns"]
                - current_summary["columns"],
                "missing_values_handled": original_summary["missing_values"]
                - current_summary["missing_values"],
                "duplicates_removed": original_summary["duplicates"]
                - current_summary["duplicates"],
            },
            "cleaning_log": self.cleaning_log,
            "data_quality_score": float(self._calculate_quality_score()),
        }

    def _calculate_quality_score(self) -> float:
        total_cells = int(self.df.shape[0]) * int(self.df.shape[1])
        if total_cells == 0:
            return 0.0
        missing_ratio = float(self.df.isnull().sum().sum()) / total_cells
        duplicate_ratio = (
            float(self.df.duplicated().sum()) / self.df.shape[0]
            if self.df.shape[0] > 0
            else 0
        )
        return max(0, min(100, 100 * (1 - missing_ratio - duplicate_ratio * 0.5)))

    # ========== CLEANING METHODS ==========
    def handle_missing_values(
        self,
        strategy: str,
        columns: Optional[List[str]] = None,
        fill_value: Optional[Any] = None,
    ):
        if columns is None:
            columns = self.df.columns.tolist()
        before_rows = len(self.df)

        try:
            if strategy == "drop":
                self.df.dropna(subset=columns, inplace=True)
            elif strategy == "fill_mean":
                for col in columns:
                    if self.df[col].dtype in ["int64", "float64"]:
                        self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == "fill_median":
                for col in columns:
                    if self.df[col].dtype in ["int64", "float64"]:
                        self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == "fill_mode":
                for col in columns:
                    mode_val = (
                        self.df[col].mode().iloc[0]
                        if not self.df[col].mode().empty
                        else None
                    )
                    self.df[col] = self.df[col].fillna(mode_val)
            elif strategy == "fill_constant":
                if fill_value is None:
                    raise ValueError(
                        "fill_value must be provided when using 'fill_constant'"
                    )
                self.df[columns] = self.df[columns].fillna(fill_value)
            else:
                raise ValueError(f"Unknown missing value strategy: {strategy}")

            self.log_action(
                "handle_missing_values",
                {
                    "strategy": strategy,
                    "columns": columns,
                    "rows_before": before_rows,
                    "rows_after": len(self.df),
                    "fill_value": fill_value,
                },
            )
        except Exception as e:
            self.log_action(
                "handle_missing_values_failed",
                {"strategy": strategy, "columns": columns, "error": str(e)},
            )
            raise

    def remove_duplicates(
        self, subset: Optional[List[str]] = None, keep: str = "first"
    ):
        before_rows = len(self.df)
        try:
            self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
            self.log_action(
                "remove_duplicates",
                {
                    "subset": subset,
                    "keep": keep,
                    "rows_before": before_rows,
                    "rows_after": len(self.df),
                },
            )
        except Exception as e:
            self.log_action(
                "remove_duplicates_failed", {"subset": subset, "error": str(e)}
            )
        raise

    def clean_text_columns(
        self,
        columns: Optional[List[str]] = None,
        operations: Optional[List[str]] = None,
    ):
        if columns is None:
            columns = self.df.select_dtypes(include=["object"]).columns.tolist()
            if operations is None:
                operations = ["strip", "lower"]

        try:
            for col in columns:
                if col not in self.df.columns:
                    continue

            # It's more efficient to work on a temporary series
            temp_series = self.df[col].astype(str)

            for op in operations:
                if op == "strip":
                    temp_series = temp_series.str.strip()
                elif op == "lower":
                    temp_series = temp_series.str.lower()
                elif op == "upper":
                    temp_series = temp_series.str.upper()
                elif op == "remove_special_chars":
                    temp_series = temp_series.str.replace(
                        r"[^a-zA-Z0-9\s]", "", regex=True
                    )
                # --- NEW: Add the missing logic for standardizing whitespace ---
                elif op == "standardize_whitespace":
                    # This regex replaces one or more space characters with a single space
                    temp_series = temp_series.str.replace(r"\s+", " ", regex=True)

            # Attempt to convert back to numeric, otherwise keep as text
            self.df[col] = pd.to_numeric(temp_series, errors="ignore")

            self.log_action(
                "clean_text_columns", {"columns": columns, "operations": operations}
            )
        except Exception as e:
            self.log_action(
                "clean_text_columns_failed", {"columns": columns, "error": str(e)}
            )
        raise

    def handle_outliers(
        self,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        action: str = "remove",
    ):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        try:
            for col in columns:
                Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

                if action == "remove":
                    self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
                elif action == "cap":
                    self.df[col] = np.where(self.df[col] < lower, lower, self.df[col])
                    self.df[col] = np.where(self.df[col] > upper, upper, self.df[col])

            self.log_action(
                "handle_outliers",
                {"columns": columns, "method": method, "action": action},
            )
        except Exception as e:
            self.log_action(
                "handle_outliers_failed", {"columns": columns, "error": str(e)}
            )
        raise

    def convert_data_types(self, conversions: Dict[str, str]):
        for col, target_type in conversions.items():
            if col not in self.df.columns:
                self.log_action(
                    "convert_types_failed",
                    {
                        "column": col,
                        "target_type": target_type,
                        "error": "Column not found",
                    },
                )
                continue
            try:
                if target_type.lower() in ["datetime", "date"]:
                    self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
                elif target_type.lower() in ["int", "integer"]:
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype(
                        "Int64"
                    )
                elif target_type.lower() in ["float", "double"]:
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                elif target_type.lower() in ["str", "string", "text"]:
                    self.df[col] = self.df[col].astype(str)
                elif target_type.lower() in ["bool", "boolean"]:
                    self.df[col] = self.df[col].astype(bool)
                else:
                    self.df[col] = self.df[col].astype(target_type)

                self.log_action(
                    "convert_types",
                    {"column": col, "target_type": target_type, "rows": len(self.df)},
                )
            except Exception as e:
                self.log_action(
                    "convert_types_failed",
                    {"column": col, "target_type": target_type, "error": str(e)},
                )

    def standardize_column_names(self):
        try:
            old_cols = self.df.columns.tolist()
            self.df.columns = [
                c.strip().lower().replace(" ", "_") for c in self.df.columns
            ]
            self.log_action(
                "standardize_column_names",
                {"old_columns": old_cols, "new_columns": self.df.columns.tolist()},
            )
        except Exception as e:
            self.log_action("standardize_column_names_failed", {"error": str(e)})
            raise

    # ========== RECOMMENDATIONS ==========
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Generate simple cleaning recommendations based on detected issues."""
        issues = self.detect_issues()
        recs = []

        for col, count in issues["missing_values"].items():
            recs.append(
                {
                    "type": "missing_values",
                    "column": col,
                    "strategy": "fill_mean",
                    "reason": f"{count} missing values detected",
                    "priority": "high",
                    "parameters": {"conversions": {}},
                }
            )

        if issues["duplicates"] > 0:
            recs.append(
                {
                    "type": "duplicates",
                    "strategy": "remove",
                    "reason": f"{issues['duplicates']} duplicate rows detected",
                    "priority": "medium",
                    "parameters": {},
                }
            )

        for col in issues["outliers"].keys():
            recs.append(
                {
                    "type": "outliers",
                    "column": col,
                    "strategy": "iqr",
                    "reason": "Potential outliers detected",
                    "priority": "medium",
                    "parameters": {},
                }
            )

        return recs
