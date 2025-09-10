# app\deep_analysis\outlier_detector.py

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Optional


class OutlierDetector:
    """
    Modular outlier detection supporting IQR, Z-score, and Isolation Forest.
    Produces both machine-friendly flags and human-readable summaries.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        methods: Optional[List[str]] = None,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        iso_forest_contamination: float = 0.05,
    ):
        self.df = df
        self.methods = methods or ["iqr"]
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.iso_forest_contamination = iso_forest_contamination

    # ------------------------------
    # Detection Methods
    # ------------------------------

    def detect_iqr(self, series: pd.Series) -> pd.Series:
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - self.iqr_multiplier * iqr, q3 + self.iqr_multiplier * iqr
        return (series < lower) | (series > upper)

    def detect_zscore(self, series: pd.Series) -> pd.Series:
        mean, std = series.mean(), series.std()
        if std == 0:  # constant column, no outliers possible
            return pd.Series(False, index=series.index)
        zscores = (series - mean) / std
        return np.abs(zscores) > self.zscore_threshold

    def detect_isolation_forest(self, series: pd.Series) -> pd.Series:
        model = IsolationForest(
            contamination=self.iso_forest_contamination,
            random_state=42,
        )
        reshaped = series.values.reshape(-1, 1)
        flags = model.fit_predict(reshaped)
        return pd.Series(flags == -1, index=series.index)

    # ------------------------------
    # Pipeline
    # ------------------------------

    def apply_pipeline(self) -> Dict[str, Dict[str, any]]:
        results: Dict[str, Dict[str, any]] = {}

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            series = self.df[col].dropna()
            if series.empty:
                continue

            method_flags = {}

            if "iqr" in self.methods:
                method_flags["iqr"] = self.detect_iqr(series)

            if "zscore" in self.methods:
                method_flags["zscore"] = self.detect_zscore(series)

            if "isoforest" in self.methods:
                method_flags["isoforest"] = self.detect_isolation_forest(series)

            # Consensus: flagged if majority of methods agree
            if len(method_flags) == 1:
                consensus_flags = next(iter(method_flags.values()))
            else:
                flag_matrix = pd.DataFrame(method_flags)
                consensus_flags = flag_matrix.mean(axis=1) > 0.5

            results[col] = {
                "flags": consensus_flags.reindex(self.df.index, fill_value=False),
                "summary": self.summarize(col, series, consensus_flags),
                "outlier_count": outlier_count,
            }

        return results

    # ------------------------------
    # Human-readable summaries
    # ------------------------------

    def summarize(self, col: str, series: pd.Series, flags: pd.Series) -> str:
        n_outliers = int(flags.sum())
        n_total = len(series)
        if n_outliers == 0:
            return f"Column '{col}': No unusual values detected."
        perc = (n_outliers / n_total) * 100
        return (
            f"Column '{col}': {n_outliers} potential outliers "
            f"({perc:.1f}% of {n_total} values)."
        )
