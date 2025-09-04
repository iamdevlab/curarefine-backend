from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from app.services.external_ai_service import generate_external_recommendations
from app.services.postgres_client import (
    get_connection,
    get_active_llm_settings_for_user,
)
from psycopg2.extras import RealDictCursor
import re
import json
from app.deep_analysis.outlier_detector import OutlierDetector
from app.services.security import get_current_user
from fastapi import Depends

# Import cleaning components
from app.api.clean import get_or_create_cleaner
from app.services.data_cleaner import DataCleaner

router = APIRouter(prefix="/ai", tags=["ai-assistance"])


class AIAnalysisRequest(BaseModel):
    file_id: str
    # user_id: int  # ✅ Add user_id
    analysis_type: str  # "suggest_cleaning", "detect_patterns", "quality_assessment", "column_analysis"
    specific_columns: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = {}
    provider: Optional[str] = None


class AutoCleanRequest(BaseModel):
    file_id: str
    # user_id: int  # ✅ Add user_id
    aggressive_level: str = "moderate"  # "conservative", "moderate", "aggressive"
    preserve_columns: Optional[List[str]] = None
    custom_rules: Optional[Dict[str, Any]] = {}


class AIInsightsRequest(BaseModel):
    file_id: str
    # user_id: int  # ✅ Add user_id
    insight_types: List[str] = [
        "data_quality",
        "patterns",
        "anomalies",
        "recommendations",
    ]


class DataIntelligenceService:
    """AI-powered data analysis and cleaning suggestions"""

    @staticmethod
    def analyze_column_patterns(series: pd.Series) -> Dict[str, Any]:
        """Analyze patterns in a column using AI-like heuristics"""

        analysis = {
            "column_name": series.name,
            "data_type": str(series.dtype),
            "unique_values": series.nunique(),
            "missing_count": series.isnull().sum(),
            "patterns": [],
            "suggestions": [],
            "confidence": 0.0,
        }

        # Convert to string for pattern analysis
        str_series = series.astype(str).str.strip()

        # Email pattern detection
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        email_matches = str_series.str.match(email_pattern, na=False).sum()
        if email_matches > len(series) * 0.7:
            analysis["patterns"].append(
                {
                    "type": "email",
                    "confidence": email_matches / len(series),
                    "description": "Column contains email addresses",
                }
            )
            analysis["suggestions"].append(
                "Consider email validation and standardization"
            )

        # Phone number pattern detection
        phone_pattern = r"^[\+]?[1-9]?[0-9]{7,15}$"
        phone_matches = (
            str_series.str.replace(r"[^\d+]", "", regex=True)
            .str.match(phone_pattern, na=False)
            .sum()
        )
        if phone_matches > len(series) * 0.6:
            analysis["patterns"].append(
                {
                    "type": "phone",
                    "confidence": phone_matches / len(series),
                    "description": "Column contains phone numbers",
                }
            )
            analysis["suggestions"].append(
                "Consider phone number formatting standardization"
            )

        # Date pattern detection
        try:
            sample = series.dropna().sample(min(10, len(series)), random_state=42)
            date_patterns = {
                "%Y-%m-%d": r"\d{4}-\d{2}-\d{2}",  # 2023-08-31
                "%d/%m/%Y": r"\d{2}/\d{2}/\d{4}",  # 31/08/2023
                "%m/%d/%Y": r"\d{2}/\d{2}/\d{4}",  # 08/31/2023 (US style)
            }

            parsed = None

            # Check if the column is mostly numeric → skip datetime test
            if pd.api.types.is_numeric_dtype(series):
                parsed = pd.Series([pd.NaT] * len(series))  # force fail datetime
            else:
                matched = False
                for fmt, regex in date_patterns.items():
                    if sample.str.contains(regex).mean() > 0.5:
                        parsed = pd.to_datetime(series, errors="coerce", format=fmt)
                        matched = True
                        break
                if not matched:
                    parsed = pd.to_datetime(series, errors="coerce")

            success_ratio = parsed.notna().mean()

            if success_ratio > 0.8:  # at least 80% look like dates
                analysis["detected_type"] = "datetime"
                analysis["confidence"] = success_ratio
                analysis["patterns"].append(
                    {
                        "type": "datetime",
                        "confidence": success_ratio,
                        "description": "Column resembles datetime format",
                    }
                )

        except Exception:
            pass

        # Numeric pattern detection
        if series.dtype == "object":
            try:
                numeric_converted = pd.to_numeric(series, errors="coerce")
                valid_numbers = numeric_converted.notna().sum()
                if valid_numbers > len(series) * 0.8:
                    analysis["patterns"].append(
                        {
                            "type": "numeric",
                            "confidence": valid_numbers / len(series),
                            "description": "Column contains numeric values stored as text",
                        }
                    )
                    analysis["suggestions"].append(
                        "Consider converting to numeric type"
                    )
            except:
                pass

        # Categorical pattern detection
        if series.nunique() < len(series) * 0.1 and series.nunique() > 1:
            analysis["patterns"].append(
                {
                    "type": "categorical",
                    "confidence": 1.0 - (series.nunique() / len(series)),
                    "description": f"Column has {series.nunique()} unique values, likely categorical",
                }
            )
            analysis["suggestions"].append(
                "Consider converting to category type for memory efficiency"
            )

        # ID pattern detection
        if series.nunique() == len(series) and not series.isnull().any():
            analysis["patterns"].append(
                {
                    "type": "unique_id",
                    "confidence": 1.0,
                    "description": "Column contains unique identifiers",
                }
            )
            analysis["suggestions"].append("This appears to be an ID column")

        # Calculate overall confidence
        if analysis["patterns"]:
            analysis["confidence"] = max(
                [p["confidence"] for p in analysis["patterns"]]
            )

        return analysis

    @staticmethod
    def detect_data_quality_issues(df: pd.DataFrame) -> Dict[str, Any]:
        """Detect comprehensive data quality issues"""

        issues = {"critical": [], "warning": [], "info": [], "overall_score": 0.0}

        total_cells = df.shape[0] * df.shape[1]

        # Critical issues
        if df.empty:
            issues["critical"].append(
                {
                    "type": "empty_dataset",
                    "description": "Dataset is empty",
                    "impact": "high",
                }
            )

        # Missing data analysis
        missing_percentage = (
            (df.isnull().sum().sum() / total_cells) * 100 if total_cells > 0 else 0.0
        )
        if missing_percentage > 50:
            issues["critical"].append(
                {
                    "type": "excessive_missing_data",
                    "description": f"{missing_percentage:.1f}% of data is missing",
                    "impact": "high",
                }
            )
        elif missing_percentage > 20:
            issues["warning"].append(
                {
                    "type": "significant_missing_data",
                    "description": f"{missing_percentage:.1f}% of data is missing",
                    "impact": "medium",
                }
            )

        # Duplicate analysis
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        if duplicate_percentage > 30:
            issues["critical"].append(
                {
                    "type": "excessive_duplicates",
                    "description": f"{duplicate_percentage:.1f}% of rows are duplicates",
                    "impact": "high",
                }
            )
        elif duplicate_percentage > 10:
            issues["warning"].append(
                {
                    "type": "significant_duplicates",
                    "description": f"{duplicate_percentage:.1f}% of rows are duplicates",
                    "impact": "medium",
                }
            )

        # Column-specific issues
        for col in df.columns:
            # Single value columns
            if df[col].nunique() <= 1:
                issues["warning"].append(
                    {
                        "type": "single_value_column",
                        "description": f"Column '{col}' has only one unique value",
                        "column": col,
                        "impact": "medium",
                    }
                )

            # High cardinality in object columns
            if df[col].dtype == "object" and df[col].nunique() > len(df) * 0.9:
                issues["info"].append(
                    {
                        "type": "high_cardinality",
                        "description": f"Column '{col}' has very high cardinality",
                        "column": col,
                        "impact": "low",
                    }
                )

        # Calculate overall quality score
        critical_weight = len(issues["critical"]) * 30
        warning_weight = len(issues["warning"]) * 10
        total_penalty = min(100, critical_weight + warning_weight)
        issues["overall_score"] = max(0, 100 - total_penalty)

        return issues

    @staticmethod
    def generate_cleaning_recommendations(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate AI-powered cleaning recommendations"""

        recommendations = []

        # Analyze each column
        for col in df.columns:
            column_analysis = DataIntelligenceService.analyze_column_patterns(df[col])

            # Missing value recommendations
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 0:
                if missing_pct < 5:
                    strategy = "drop_rows"
                    reason = "This column has very few missing values, so the affected rows can be safely removed."
                elif df[col].dtype in ["int64", "float64"]:
                    strategy = "fill_median"
                    reason = "This is a numeric column; filling with the median is a robust choice."
                elif missing_pct < 30:
                    strategy = "fill_mode"
                    reason = "This column has repeating text; filling with the most common value is a good option."
                else:
                    strategy = "drop_columns"
                    reason = "Over 30% of the values in this column are missing; consider removing the column entirely."

                recommendations.append(
                    {
                        "type": "missing_values",
                        "column": col,
                        "strategy": strategy,
                        "reason": reason,
                        "priority": "high" if missing_pct > 20 else "medium",
                        "parameters": {"columns": [col], "strategy": strategy},
                    }
                )

            # Data type conversion recommendations
            for pattern in column_analysis["patterns"]:
                if pattern["type"] == "numeric" and df[col].dtype == "object":
                    recommendations.append(
                        {
                            "type": "convert_types",
                            "column": col,
                            "strategy": "convert_to_numeric",
                            "reason": "This column contains numbers that are stored as text and should be converted.",
                            "priority": "medium",
                            "parameters": {"conversions": {col: "float"}},
                        }
                    )
                elif pattern["type"] == "date" and df[col].dtype == "object":
                    recommendations.append(
                        {
                            "type": "convert_types",
                            "column": col,
                            "strategy": "convert_to_datetime",
                            "reason": "This column appears to contain dates and should be converted for proper analysis.",
                            "priority": "medium",
                            "parameters": {"conversions": {col: "datetime"}},
                        }
                    )
                elif pattern["type"] == "categorical":
                    recommendations.append(
                        {
                            "type": "convert_types",
                            "column": col,
                            "strategy": "convert_to_category",
                            "reason": "This column contains many repeating values and can be optimized for better performance.",
                            "priority": "low",
                            "parameters": {"conversions": {col: "category"}},
                        }
                    )

        # Global recommendations
        if df.duplicated().sum() > 0:
            recommendations.append(
                {
                    "type": "duplicates",
                    "strategy": "remove_duplicates",
                    "reason": f"Found {df.duplicated().sum()} duplicate rows that should be removed.",
                    "priority": "high",
                    "parameters": {"keep": "first"},
                }
            )

        # Text cleaning recommendations
        text_columns = df.select_dtypes(include=["object"]).columns.tolist()
        for (
            col
        ) in (
            text_columns
        ):  # Changed from a single global recommendation to a targeted one
            if (
                df[col]
                .astype(str)
                .str.contains(r"^\s+|\s+$", regex=True, na=False)
                .any()
            ):
                recommendations.append(
                    {
                        "type": "text_cleaning",
                        "column": col,
                        "strategy": "clean_text_columns",
                        "reason": f"Column '{col}' has entries with extra spaces that should be removed.",
                        "priority": "medium",
                        "parameters": {
                            "columns": [col],  # Now targets only the specific column
                            "operations": ["strip", "standardize_whitespace"],
                        },
                    }
                )

        return recommendations


@router.post("/analyze")
async def ai_analysis(
    request: AIAnalysisRequest, current_user: dict = Depends(get_current_user)
):
    """Perform AI-powered data analysis, choosing the engine dynamically based on the request."""
    user_id = current_user["user_id"]
    try:
        cleaner = get_or_create_cleaner(user_id, request.file_id)
        df = cleaner.df

        result = {
            "file_id": request.file_id,
            "analysis_type": request.analysis_type,
            "timestamp": datetime.now().isoformat(),
        }

        use_external_ai = request.provider and request.provider != "curarefine_analyzer"

        if request.analysis_type == "suggest_cleaning":
            recommendations = []
            analysis_engine = "CuraRefine Analyzer"

            if use_external_ai:
                print(
                    f"--- Frontend requested external provider: {request.provider} ---"
                )
                active_settings = None
                with get_connection() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        active_settings = get_active_llm_settings_for_user(
                            user_id, cursor
                        )

                if active_settings and (
                    active_settings.get("api_key")
                    or active_settings.get("self_hosted_endpoint")
                ):
                    print("--- Found valid settings. Using External AI Service. ---")
                    analysis_engine = active_settings.get("provider", "external")
                    recommendations = await generate_external_recommendations(
                        df, active_settings
                    )
                else:
                    print(
                        f"--- Could not find valid settings for {request.provider}. Falling back to internal engine. ---"
                    )
                    recommendations = (
                        DataIntelligenceService.generate_cleaning_recommendations(df)
                    )
            else:
                print("--- Using Internal DataIntelligenceService as requested. ---")
                recommendations = (
                    DataIntelligenceService.generate_cleaning_recommendations(df)
                )

            result["analysis_engine"] = analysis_engine
            result["recommendations"] = recommendations
            result["total_recommendations"] = len(recommendations)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Analysis type '{request.analysis_type}' not supported by this endpoint.",
            )

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing AI analysis: {str(e)}"
        )


@router.post("/deep-analysis")
async def deep_data_analysis(
    request: AIAnalysisRequest, current_user: dict = Depends(get_current_user)
):
    """
    Performs a comprehensive analysis, dynamically choosing between the
    internal deep scan (rules + outliers) and an external AI provider.
    """
    user_id = current_user["user_id"]
    try:
        # 1. Get the current DataFrame
        cleaner = get_or_create_cleaner(user_id, request.file_id)
        df = cleaner.df

        recommendations = []
        analysis_engine = "CuraRefine Analyzer"  # Default to internal

        # --- START: NEW DYNAMIC LOGIC ---
        use_external_ai = request.provider and request.provider != "curarefine_analyzer"

        if use_external_ai:
            # 🤖 Frontend requested an external provider for the deep scan.
            print(
                f"--- Deep Scan requested with external provider: {request.provider} ---"
            )
            active_settings = None
            with get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    active_settings = get_active_llm_settings_for_user(user_id, cursor)

            if active_settings and (
                active_settings.get("api_key")
                or active_settings.get("self_hosted_endpoint")
            ):
                print(
                    "--- Found valid settings. Using External AI Service for Deep Scan. ---"
                )
                analysis_engine = active_settings.get("provider", "external")
                # For deep scan, we can reuse the same external recommendation function
                recommendations = await generate_external_recommendations(
                    df, active_settings
                )
            else:
                print(
                    f"--- Could not find settings for {request.provider}. Falling back to internal deep scan. ---"
                )
                # Fallback logic is the same as the 'else' block below
                initial_recommendations = (
                    DataIntelligenceService.generate_cleaning_recommendations(df)
                )
                outlier_detector = OutlierDetector(df=df, methods=["iqr", "zscore"])
                outlier_results = outlier_detector.apply_pipeline()
                outlier_recommendations = [
                    {
                        "type": "outliers",
                        "column": col,
                        "strategy": "handle_outliers",
                        "reason": result["summary"],
                        "priority": "medium",
                        "parameters": {
                            "columns": [col],
                            "method": "iqr",
                            "action": "remove",
                        },
                    }
                    for col, result in outlier_results.items()
                    if "potential outliers" in result["summary"]
                ]
                recommendations = initial_recommendations + outlier_recommendations

        else:
            # ⚙️ Frontend requested the internal deep scan.
            # This combines the standard recommendations with outlier detection.
            print("--- Running Internal Deep Scan (Rules + Outliers) as requested. ---")

            # Run the initial internal analysis to get baseline recommendations
            initial_recommendations = (
                DataIntelligenceService.generate_cleaning_recommendations(df)
            )

            # Run the advanced outlier analysis
            outlier_detector = OutlierDetector(df=df, methods=["iqr", "zscore"])
            outlier_results = outlier_detector.apply_pipeline()

            # Convert the outlier results into the standard recommendation format
            outlier_recommendations = [
                {
                    "type": "outliers",
                    "column": col,
                    "strategy": "handle_outliers",
                    "reason": result["summary"],
                    "priority": "medium",
                    "parameters": {
                        "columns": [col],
                        "method": "iqr",
                        "action": "remove",
                    },
                }
                for col, result in outlier_results.items()
                if "potential outliers" in result["summary"]
            ]

            # Combine both lists into a single response
            recommendations = initial_recommendations + outlier_recommendations

        # --- END: NEW DYNAMIC LOGIC ---

        return JSONResponse(
            {
                "file_id": request.file_id,
                "analysis_type": "deep_analysis",
                "analysis_engine": analysis_engine,
                "timestamp": datetime.now().isoformat(),
                "recommendations": recommendations,
                "total_recommendations": len(recommendations),
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing deep analysis: {str(e)}"
        )


# Main endpoint for AI-powered data analysis


# @router.post("/analyze")
# async def ai_analysis(
#     request: AIAnalysisRequest, current_user: dict = Depends(get_current_user)
# ):
#     """Perform AI-powered data analysis, choosing the engine dynamically."""
#     user_id = current_user["user_id"]
#     try:
#         cleaner = get_or_create_cleaner(user_id, request.file_id)
#         df = cleaner.df

#         result = {
#             "file_id": request.file_id,
#             "analysis_type": request.analysis_type,
#             "timestamp": datetime.now().isoformat(),
#         }

#         # --- NEW: Check for user's external AI settings ---
#         active_settings = None
#         with get_connection() as conn:
#             with conn.cursor(cursor_factory=RealDictCursor) as cursor:
#                 active_settings = get_active_llm_settings_for_user(user_id, cursor)
#         print(f"--- 🔍 Retrieved AI Settings for user {user_id}: {active_settings} ---")

#         if request.analysis_type == "suggest_cleaning":

#             recommendations = []
#             analysis_engine = "internal"  # Default to the built-in engine

#             # --- MODIFIED: Dynamic switch for generating recommendations ---
#             if active_settings and (
#                 active_settings.get("api_key")
#                 or active_settings.get("self_hosted_endpoint")
#             ):
#                 # 🤖 If settings are found, use the new external AI service.
#                 print("--- Using External AI Service ---")
#                 analysis_engine = active_settings.get("provider", "external")
#                 recommendations = await generate_external_recommendations(
#                     df, active_settings
#                 )
#             else:
#                 # ⚙️ If no settings are found, run the original, working code as a fallback.
#                 print("--- Using Internal DataIntelligenceService (Fallback) ---")
#                 recommendations = (
#                     DataIntelligenceService.generate_cleaning_recommendations(df)
#                 )

#             # --- NEW: Add the analysis engine to the result ---
#             result["analysis_engine"] = analysis_engine
#             result["recommendations"] = recommendations
#             result["total_recommendations"] = len(recommendations)

#         elif request.analysis_type == "detect_patterns":
#             columns_to_analyze = request.specific_columns or df.columns.tolist()
#             patterns = {}
#             for col in columns_to_analyze:
#                 if col in df.columns:
#                     patterns[col] = DataIntelligenceService.analyze_column_patterns(
#                         df[col]
#                     )
#             result["patterns"] = patterns

#         elif request.analysis_type == "quality_assessment":
#             quality_issues = DataIntelligenceService.detect_data_quality_issues(df)
#             result["quality_assessment"] = quality_issues

#         elif request.analysis_type == "column_analysis":
#             columns_to_analyze = request.specific_columns or df.columns.tolist()
#             column_insights = {}
#             for col in columns_to_analyze:
#                 if col in df.columns:
#                     insights = {
#                         "basic_stats": {
#                             "count": len(df[col]),
#                             "unique": df[col].nunique(),
#                             "missing": df[col].isnull().sum(),
#                             "data_type": str(df[col].dtype),
#                         },
#                         "patterns": DataIntelligenceService.analyze_column_patterns(
#                             df[col]
#                         ),
#                     }
#                     if df[col].dtype in ["int64", "float64"]:
#                         insights["numeric_stats"] = {
#                             "mean": (
#                                 float(df[col].mean())
#                                 if not df[col].isnull().all()
#                                 else None
#                             ),
#                             "median": (
#                                 float(df[col].median())
#                                 if not df[col].isnull().all()
#                                 else None
#                             ),
#                             "std": (
#                                 float(df[col].std())
#                                 if not df[col].isnull().all()
#                                 else None
#                             ),
#                             "min": (
#                                 float(df[col].min())
#                                 if not df[col].isnull().all()
#                                 else None
#                             ),
#                             "max": (
#                                 float(df[col].max())
#                                 if not df[col].isnull().all()
#                                 else None
#                             ),
#                         }
#                     column_insights[col] = insights
#             result["column_insights"] = column_insights

#         else:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Unknown analysis type: {request.analysis_type}",
#             )

#         return JSONResponse(result)

#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Error performing AI analysis: {str(e)}"
#         )


# New endpoint for deep data analysis combining multiple techniques
# @router.post("/deep-analysis")
# async def deep_data_analysis(
#     request: AIAnalysisRequest, current_user: dict = Depends(get_current_user)
# ):
#     """
#     Performs a comprehensive analysis including internal recommendations
#     and advanced outlier detection.
#     """
#     user_id = current_user["user_id"]
#     try:
#         # 1. Get the current, cleaned DataFrame for the user's session
#         cleaner = get_or_create_cleaner(user_id, request.file_id)
#         df = cleaner.df

#         # 2. Run the initial internal analysis to get baseline recommendations
#         print("--- Running Internal DataIntelligenceService for Deep Scan ---")
#         initial_recommendations = (
#             DataIntelligenceService.generate_cleaning_recommendations(df)
#         )

#         # 3. Run the advanced outlier analysis
#         print("--- Running OutlierDetector for Deep Scan ---")
#         outlier_detector = OutlierDetector(df=df, methods=["iqr", "zscore"])
#         outlier_results = outlier_detector.apply_pipeline()

#         # 4. Convert the outlier results into the standard recommendation format
#         outlier_recommendations = []
#         for col, result in outlier_results.items():
#             # Only add a recommendation if the summary indicates outliers were found
#             if "potential outliers" in result["summary"]:
#                 outlier_recommendations.append(
#                     {
#                         "type": "outliers",
#                         "column": col,
#                         "strategy": "handle_outliers",  # Suggests which cleaner function to use
#                         "reason": result["summary"],
#                         "priority": "medium",
#                         "parameters": {
#                             "columns": [col],
#                             "method": "iqr",
#                             "action": "remove",
#                         },
#                     }
#                 )

#         # 5. Combine both lists into a single response
#         combined_recommendations = initial_recommendations + outlier_recommendations

#         return JSONResponse(
#             {
#                 "file_id": request.file_id,
#                 "analysis_type": "deep_analysis",
#                 "analysis_engine": "DataCura Analyzer",
#                 "timestamp": datetime.now().isoformat(),
#                 "recommendations": combined_recommendations,
#                 "total_recommendations": len(combined_recommendations),
#             }
#         )

#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Error performing deep analysis: {str(e)}"
#         )


# New endpoint for automated cleaning based on AI recommendations
@router.post("/auto-clean")
async def auto_clean_data(
    request: AutoCleanRequest, current_user: dict = Depends(get_current_user)
):
    """Automatically clean data using AI recommendations"""
    user_id = current_user["user_id"]
    try:
        cleaner = get_or_create_cleaner(user_id, request.file_id)

        # Generate recommendations
        recommendations = DataIntelligenceService.generate_cleaning_recommendations(
            cleaner.df
        )

        # Filter recommendations based on aggressiveness level
        filtered_recommendations = []

        for rec in recommendations:
            if request.aggressive_level == "conservative":
                if rec["priority"] == "high" and rec["type"] in [
                    "duplicates",
                    "missing_values",
                ]:
                    filtered_recommendations.append(rec)
            elif request.aggressive_level == "moderate":
                if rec["priority"] in ["high", "medium"]:
                    filtered_recommendations.append(rec)
            elif request.aggressive_level == "aggressive":
                filtered_recommendations.append(rec)

        # Apply cleaning steps
        applied_steps = []
        errors = []

        for rec in filtered_recommendations:
            # Skip if column is in preserve list
            if (
                request.preserve_columns
                and rec.get("column") in request.preserve_columns
            ):
                continue

            try:
                params = rec["parameters"]

                if rec["type"] == "missing_values":
                    cleaner.handle_missing_values(**params)
                elif rec["type"] == "duplicates":
                    cleaner.remove_duplicates(**params)
                elif rec["type"] == "convert_types":
                    cleaner.convert_data_types(**params)
                elif rec["type"] == "text_cleaning":
                    cleaner.clean_text_columns(**params)

                applied_steps.append({"recommendation": rec, "status": "success"})

            except Exception as step_error:
                errors.append({"recommendation": rec, "error": str(step_error)})

        # Get final report
        cleaning_report = cleaner.get_cleaning_report()

        return JSONResponse(
            {
                "status": "completed",
                "file_id": request.file_id,
                "aggressive_level": request.aggressive_level,
                "total_recommendations": len(recommendations),
                "applied_steps": len(applied_steps),
                "errors": len(errors),
                "applied_steps_details": applied_steps,
                "errors_details": errors,
                "cleaning_report": cleaning_report,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in auto-clean: {str(e)}")


# New endpoint to get AI-powered insights
@router.post("/insights")
async def get_ai_insights(
    request: AIInsightsRequest, current_user: dict = Depends(get_current_user)
):
    """Get comprehensive AI-powered insights about the data"""
    user_id = current_user["user_id"]
    try:
        cleaner = get_or_create_cleaner(user_id, request.file_id)
        df = cleaner.df

        insights = {
            "file_id": request.file_id,
            "timestamp": datetime.now().isoformat(),
            "dataset_overview": {
                "shape": df.shape,
                "memory_usage": df.memory_usage(deep=True).sum(),
                "data_types_distribution": df.dtypes.value_counts().to_dict(),
            },
        }

        if "data_quality" in request.insight_types:
            insights["data_quality"] = (
                DataIntelligenceService.detect_data_quality_issues(df)
            )

        if "patterns" in request.insight_types:
            # Analyze patterns in all columns
            patterns = {}
            for col in df.columns:
                patterns[col] = DataIntelligenceService.analyze_column_patterns(df[col])
            insights["patterns"] = patterns

        if "anomalies" in request.insight_types:
            anomalies = []

            # Detect anomalies in numeric columns
            for col in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[
                    (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                ]

                if len(outliers) > 0:
                    anomalies.append(
                        {
                            "column": col,
                            "type": "statistical_outliers",
                            "count": len(outliers),
                            "percentage": (len(outliers) / len(df)) * 100,
                            "method": "IQR",
                        }
                    )

            insights["anomalies"] = anomalies

        if "recommendations" in request.insight_types:
            insights["recommendations"] = (
                DataIntelligenceService.generate_cleaning_recommendations(df)
            )

        return JSONResponse(insights)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating insights: {str(e)}"
        )


# New endpoint to get AI capabilities and options
@router.get("/capabilities")
async def get_ai_capabilities():
    """Get information about AI capabilities"""

    return JSONResponse(
        {
            "analysis_types": {
                "suggest_cleaning": "Generate automated cleaning recommendations",
                "detect_patterns": "Identify data patterns and formats",
                "quality_assessment": "Assess overall data quality",
                "column_analysis": "Detailed analysis of individual columns",
            },
            "auto_clean_levels": {
                "conservative": "Only apply high-priority, safe cleaning operations",
                "moderate": "Apply high and medium priority cleaning operations",
                "aggressive": "Apply all recommended cleaning operations",
            },
            "insight_types": {
                "data_quality": "Overall data quality assessment",
                "patterns": "Pattern detection in all columns",
                "anomalies": "Anomaly and outlier detection",
                "recommendations": "Cleaning recommendations",
            },
            "supported_patterns": [
                "email",
                "phone",
                "date",
                "numeric",
                "categorical",
                "unique_id",
            ],
        }
    )


# New endpoint to validate a specific cleaning recommendation
@router.post("/validate-recommendation")
async def validate_cleaning_recommendation(
    file_id: str, recommendation: Dict[str, Any]
):
    """Validate a specific cleaning recommendation before applying"""

    try:
        cleaner = get_or_create_cleaner(file_id)
        df = cleaner.df.copy()  # Work with a copy

        validation_result = {
            "valid": True,
            "warnings": [],
            "estimated_impact": {},
            "recommendation": recommendation,
        }

        rec_type = recommendation.get("type")
        params = recommendation.get("parameters", {})

        if rec_type == "missing_values":
            column = params.get("columns", [])
            if column:
                missing_count = (
                    df[column[0]].isnull().sum() if column[0] in df.columns else 0
                )
                validation_result["estimated_impact"]["rows_affected"] = missing_count

                if (
                    params.get("strategy") == "drop_rows"
                    and missing_count > len(df) * 0.5
                ):
                    validation_result["warnings"].append(
                        "Dropping rows will remove more than 50% of data"
                    )

        elif rec_type == "duplicates":
            duplicate_count = df.duplicated().sum()
            validation_result["estimated_impact"]["rows_to_remove"] = duplicate_count

        elif rec_type == "convert_types":
            conversions = params.get("conversions", {})
            for col, target_type in conversions.items():
                if col in df.columns:
                    if target_type == "numeric":
                        try:
                            converted = pd.to_numeric(df[col], errors="coerce")
                            # Calculate only net new nulls from conversion errors
                            original_nulls = df[col].isnull().sum()
                            new_nulls = converted.isnull().sum()
                            failed_conversions = new_nulls - original_nulls
                            if failed_conversions > 0:
                                validation_result["warnings"].append(
                                    f"Converting {col} to numeric will create {failed_conversions} NaN values"
                                )
                        except Exception as conv_error:
                            validation_result["valid"] = False
                            validation_result["warnings"].append(
                                f"Cannot convert {col} to numeric"
                            )

        return JSONResponse(validation_result)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error validating recommendation: {str(e)}"
        )
