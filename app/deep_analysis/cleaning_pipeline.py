# app\deep_analysis\cleaning_pipeline.py
# datacura_pipeline_with_nl_suggestions.py

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from app.services.data_cleaner import DataCleaner
from app.deep_analysis.outlier_detector import OutlierDetector
import logging

logger = logging.getLogger(__name__)


class DatacuraPipeline:
    """
    Advanced data cleaning pipeline with automated recommendations and
    LLM-style natural language suggestions.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaner = DataCleaner(self.df)
        self.outlier_results: Dict[str, Any] = {}
        self.recommendations: List[Dict[str, Any]] = []
        self.nl_suggestions: List[str] = []

    # -------------------------
    # Step 1: Audit & Summarize
    # -------------------------
    def audit_data(self) -> Dict[str, Any]:
        summary = self.cleaner.get_data_summary()
        issues = self.cleaner.detect_issues()
        logger.info("Data audit completed")
        return {"summary": summary, "issues": issues}

    # -------------------------
    # Step 2: Outlier Handling
    # -------------------------
    def handle_outliers(
        self, methods: Optional[List[str]] = None, action: str = "remove"
    ):
        od = OutlierDetector(self.cleaner.df, methods=methods)
        self.outlier_results = od.apply_pipeline()

        for col, res in self.outlier_results.items():
            flags = res["flags"]
            if action == "remove":
                self.cleaner.df = self.cleaner.df[~flags]
            elif action == "cap":
                series = self.cleaner.df[col]
                lower = series[~flags].min()
                upper = series[~flags].max()
                self.cleaner.df[col] = np.where(series < lower, lower, series)
                self.cleaner.df[col] = np.where(series > upper, upper, series)

        self.cleaner.log_action(
            "handle_outliers", {"methods": methods, "action": action}
        )

    # -------------------------
    # Step 3: Missing Value Handling
    # -------------------------
    def handle_missing_values(
        self, strategy: str = "fill_mean", columns: Optional[List[str]] = None
    ):
        self.cleaner.handle_missing_values(strategy=strategy, columns=columns)

    # -------------------------
    # Step 4: Duplicates & Consistency
    # -------------------------
    def remove_duplicates(self, subset: Optional[List[str]] = None):
        self.cleaner.remove_duplicates(subset=subset)

    # -------------------------
    # Step 5: Text & Categorical Cleaning
    # -------------------------
    def clean_text_columns(
        self,
        columns: Optional[List[str]] = None,
        operations: Optional[List[str]] = None,
    ):
        self.cleaner.clean_text_columns(columns=columns, operations=operations)

    # -------------------------
    # Step 6: Type Enforcement & Validation
    # -------------------------
    def enforce_types(self, conversions: Dict[str, str]):
        self.cleaner.convert_data_types(conversions)

    # -------------------------
    # Step 7: Normalization & Scaling
    # -------------------------
    def normalize_columns(
        self, columns: Optional[List[str]] = None, method: str = "zscore"
    ):
        if columns is None:
            columns = self.cleaner.df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
        for col in columns:
            if method == "zscore":
                mean, std = self.cleaner.df[col].mean(), self.cleaner.df[col].std()
                self.cleaner.df[col] = (self.cleaner.df[col] - mean) / std
            elif method == "minmax":
                min_val, max_val = (
                    self.cleaner.df[col].min(),
                    self.cleaner.df[col].max(),
                )
                self.cleaner.df[col] = (self.cleaner.df[col] - min_val) / (
                    max_val - min_val
                )
        self.cleaner.log_action(
            "normalize_columns", {"columns": columns, "method": method}
        )

    # -------------------------
    # Step 8: Feature Engineering & Flags
    # -------------------------
    def add_flags(self):
        for col in self.cleaner.df.columns:
            if self.cleaner.df[col].isnull().sum() > 0:
                self.cleaner.df[f"{col}_missing_flag"] = self.cleaner.df[col].isnull()
        self.cleaner.log_action("add_flags", {"columns": list(self.cleaner.df.columns)})

    # -------------------------
    # Step 9: Recommendations
    # -------------------------
    def generate_recommendations(self):
        issues = self.cleaner.detect_issues()
        recs = []

        # Missing values
        for col, count in issues.get("missing_values", {}).items():
            strategy = (
                "fill_median"
                if self.cleaner.df[col].dtype in ["int64", "float64"]
                else "fill_mode"
            )
            recs.append(
                {
                    "type": "missing_values",
                    "column": col,
                    "strategy": strategy,
                    "reason": f"{count} missing values detected",
                    "priority": "high",
                }
            )

        # Duplicates
        if issues.get("duplicates", 0) > 0:
            recs.append(
                {
                    "type": "duplicates",
                    "strategy": "remove",
                    "reason": f"{issues['duplicates']} duplicate rows detected",
                    "priority": "medium",
                }
            )

        # Outliers
        for col in self.outlier_results.keys():
            recs.append(
                {
                    "type": "outliers",
                    "column": col,
                    "strategy": "iqr",
                    "reason": "Potential outliers detected",
                    "priority": "medium",
                }
            )

        # Whitespace / text
        for col, ws in issues.get("whitespace_issues", {}).items():
            recs.append(
                {
                    "type": "whitespace",
                    "column": col,
                    "strategy": "strip",
                    "reason": f"{ws.get('leading',0)} leading and {ws.get('trailing',0)} trailing spaces",
                    "priority": "low",
                }
            )

        self.recommendations = recs
        return recs

    # -------------------------
    # Step 10: Generate Natural Language Suggestions
    # -------------------------
    def generate_nl_suggestions(self):
        suggestions = []
        for rec in self.recommendations:
            col = rec.get("column", "")
            typ = rec.get("type", "")
            strat = rec.get("strategy", "")
            reason = rec.get("reason", "")
            priority = rec.get("priority", "")
            if typ == "missing_values":
                suggestions.append(
                    f"Column '{col}' has missing values. Consider using '{strat}' strategy. Reason: {reason}. Priority: {priority}."
                )
            elif typ == "duplicates":
                suggestions.append(
                    f"There are duplicate rows in the dataset. Recommended action: '{strat}'. Reason: {reason}. Priority: {priority}."
                )
            elif typ == "outliers":
                suggestions.append(
                    f"Column '{col}' may contain outliers. Suggested method: '{strat}'. Reason: {reason}. Priority: {priority}."
                )
            elif typ == "whitespace":
                suggestions.append(
                    f"Column '{col}' has whitespace issues. Suggested action: '{strat}'. Reason: {reason}. Priority: {priority}."
                )
        self.nl_suggestions = suggestions
        return suggestions

    # -------------------------
    # Step 11: Logging & Reporting
    # -------------------------
    def generate_report(self) -> Dict[str, Any]:
        report = self.cleaner.get_cleaning_report()
        self.generate_recommendations()
        report["recommendations"] = self.recommendations
        report["nl_suggestions"] = self.generate_nl_suggestions()
        return report

    # -------------------------
    # Step 12: Run Full Pipeline
    # -------------------------
    def run_pipeline(
        self,
        outlier_methods: Optional[List[str]] = None,
        outlier_action: str = "remove",
        missing_strategy: str = "fill_mean",
        text_operations: Optional[List[str]] = None,
        type_conversions: Optional[Dict[str, str]] = None,
        normalization_method: str = "zscore",
    ) -> Dict[str, Any]:
        self.audit_data()
        self.handle_outliers(methods=outlier_methods, action=outlier_action)
        self.handle_missing_values(strategy=missing_strategy)
        self.remove_duplicates()
        self.clean_text_columns(operations=text_operations)
        if type_conversions:
            self.enforce_types(type_conversions)
        self.normalize_columns(method=normalization_method)
        self.add_flags()
        return self.generate_report()
