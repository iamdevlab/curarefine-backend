# # app/services/visualizer.py
# app/services/visualizer.py

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re

# Import Plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class VisualizationService:
    """
    An enhanced service class to handle the logic of generating interactive chart specifications
    with support for multiple chart types and domain-specific visualizations.
    """

    # Domain-specific configurations
    DOMAIN_CONFIGS = {
        "finance": {
            "primary_color": "#2E86AB",
            "secondary_color": "#A23B72",
            "default_charts": ["line", "candlestick", "correlation"],
        },
        "healthcare": {
            "primary_color": "#3C91E6",
            "secondary_color": "#FA824C",
            "default_charts": ["bar", "scatter", "box"],
        },
        "retail": {
            "primary_color": "#F26419",
            "secondary_color": "#2F4858",
            "default_charts": ["bar", "pie", "treemap"],
        },
        "manufacturing": {
            "primary_color": "#86BBD8",
            "secondary_color": "#33658A",
            "default_charts": ["line", "histogram", "scatter"],
        },
        "generic": {
            "primary_color": "#636EFA",
            "secondary_color": "#EF553B",
            "default_charts": ["bar", "scatter", "histogram"],
        },
    }

    @staticmethod
    def generate_visualizations(
        table_data: List[Dict[str, Any]],
        domain: str = "generic",
        max_charts: int = 10,
        chart_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Takes table data, generates interactive charts using Plotly,
        and returns a list of chart specifications.

        Args:
            table_data: A list of dictionaries, where each dictionary represents a row.
            domain: The industry domain for domain-specific visualizations.
            max_charts: Maximum number of charts to generate.
            chart_types: Specific chart types to generate (if None, uses domain defaults).

        Returns:
            A list of dictionaries containing chart specifications.
        """
        if not table_data:
            logging.warning("Visualization request received with no data.")
            return []

        try:
            # 1. Convert data into a pandas DataFrame
            df = pd.DataFrame(table_data)
            logging.info(
                f"Successfully created DataFrame with {df.shape[0]} rows and {df.shape[1]} columns."
            )

            # 2. Clean and preprocess data
            df = VisualizationService._clean_data(df)

            # 3. Analyze data structure
            column_analysis = VisualizationService._analyze_columns(df)

            # 4. Get domain configuration
            domain_config = VisualizationService.DOMAIN_CONFIGS.get(
                domain, VisualizationService.DOMAIN_CONFIGS["generic"]
            )

            # 5. Determine which chart types to generate
            if chart_types is None:
                chart_types = domain_config["default_charts"]

            # 6. Generate charts based on data and requested types
            chart_specs = []
            charts_generated = 0

            for chart_type in chart_types:
                if charts_generated >= max_charts:
                    break

                charts = VisualizationService._generate_chart_type(
                    df, chart_type, column_analysis, domain_config
                )

                for chart in charts:
                    if charts_generated < max_charts:
                        chart_specs.append(chart)
                        charts_generated += 1
                    else:
                        break

            logging.info(f"Generated {len(chart_specs)} chart specifications.")
            return chart_specs

        except Exception as e:
            logging.error(f"An error occurred during visualization: {e}", exc_info=True)
            return []

    @staticmethod
    def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data for visualization."""
        # Make a copy to avoid modifying the original
        df_clean = df.copy()

        # Convert potential date columns
        for col in df_clean.columns:
            # Try to convert to datetime
            try:
                df_clean[col] = pd.to_datetime(df_clean[col])
            except (ValueError, TypeError):
                pass

            # Convert string numbers to numeric
            if df_clean[col].dtype == "object":
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col])
                except (ValueError, TypeError):
                    pass

        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)

        # For categorical columns, fill with mode or "Unknown"
        categorical_cols = df_clean.select_dtypes(
            include=["object", "category"]
        ).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "Unknown"
                df_clean[col].fillna(fill_val, inplace=True)

        return df_clean

    @staticmethod
    def _analyze_columns(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze column types and properties for chart generation."""
        analysis = {
            "numeric_cols": [],
            "categorical_cols": [],
            "datetime_cols": [],
            "high_cardinality_cols": [],  # Columns with too many unique values
            "low_cardinality_cols": [],  # Columns with few unique values
            "column_stats": {},
        }

        for col in df.columns:
            col_type = str(df[col].dtype)
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()

            analysis["column_stats"][col] = {
                "dtype": col_type,
                "unique_count": unique_count,
                "null_count": null_count,
                "sample_values": (
                    df[col].dropna().head(3).tolist() if unique_count > 0 else []
                ),
            }

            # Categorize columns
            if pd.api.types.is_numeric_dtype(df[col]):
                analysis["numeric_cols"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                analysis["datetime_cols"].append(col)
            else:
                analysis["categorical_cols"].append(col)

                # Check cardinality
                if unique_count > 50:
                    analysis["high_cardinality_cols"].append(col)
                elif unique_count <= 10:
                    analysis["low_cardinality_cols"].append(col)

        return analysis

    @staticmethod
    def _generate_chart_type(
        df: pd.DataFrame,
        chart_type: str,
        analysis: Dict[str, Any],
        domain_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate charts of a specific type based on data analysis."""
        charts = []

        if chart_type == "bar":
            charts.extend(
                VisualizationService._generate_bar_charts(df, analysis, domain_config)
            )
        elif chart_type == "line":
            charts.extend(
                VisualizationService._generate_line_charts(df, analysis, domain_config)
            )
        elif chart_type == "scatter":
            charts.extend(
                VisualizationService._generate_scatter_plots(
                    df, analysis, domain_config
                )
            )
        elif chart_type == "histogram":
            charts.extend(
                VisualizationService._generate_histograms(df, analysis, domain_config)
            )
        elif chart_type == "box":
            charts.extend(
                VisualizationService._generate_box_plots(df, analysis, domain_config)
            )
        elif chart_type == "violin":
            charts.extend(
                VisualizationService._generate_violin_plots(df, analysis, domain_config)
            )
        elif chart_type == "pie":
            charts.extend(
                VisualizationService._generate_pie_charts(df, analysis, domain_config)
            )
        elif chart_type == "heatmap":
            charts.extend(
                VisualizationService._generate_heatmaps(df, analysis, domain_config)
            )
        elif chart_type == "correlation":
            charts.extend(
                VisualizationService._generate_correlation_matrix(
                    df, analysis, domain_config
                )
            )
        elif chart_type == "treemap":
            charts.extend(
                VisualizationService._generate_treemaps(df, analysis, domain_config)
            )
        elif chart_type == "sunburst":
            charts.extend(
                VisualizationService._generate_sunburst_charts(
                    df, analysis, domain_config
                )
            )
        elif chart_type == "funnel":
            charts.extend(
                VisualizationService._generate_funnel_charts(
                    df, analysis, domain_config
                )
            )
        elif chart_type == "candlestick" and len(analysis["datetime_cols"]) > 0:
            charts.extend(
                VisualizationService._generate_candlestick_charts(
                    df, analysis, domain_config
                )
            )
        elif chart_type == "area":
            charts.extend(
                VisualizationService._generate_area_charts(df, analysis, domain_config)
            )
        elif chart_type == "bubble":
            charts.extend(
                VisualizationService._generate_bubble_charts(
                    df, analysis, domain_config
                )
            )
        elif chart_type == "radar":
            charts.extend(
                VisualizationService._generate_radar_charts(df, analysis, domain_config)
            )

        return charts

    @staticmethod
    def _generate_bar_charts(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate bar charts."""
        charts = []
        numeric_cols = analysis["numeric_cols"]
        categorical_cols = analysis["categorical_cols"]

        # Simple bar charts (one categorical, one numeric)
        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                if df[cat_col].nunique() <= 20:  # Avoid too many categories
                    try:
                        fig = px.bar(
                            df,
                            x=cat_col,
                            y=num_col,
                            title=f"{num_col} by {cat_col}",
                            color_discrete_sequence=[domain_config["primary_color"]],
                        )
                        charts.append(
                            {
                                "type": "bar",
                                "title": f"{num_col} by {cat_col}",
                                "spec": fig.to_json(),
                                "description": f"Bar chart showing {num_col} values across different {cat_col} categories",
                            }
                        )
                    except Exception as e:
                        logging.warning(
                            f"Failed to create bar chart for {cat_col} and {num_col}: {e}"
                        )

        # Grouped bar charts
        if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
            try:
                cat_col1, cat_col2 = categorical_cols[0], categorical_cols[1]
                num_col = numeric_cols[0]

                if df[cat_col1].nunique() <= 10 and df[cat_col2].nunique() <= 10:
                    fig = px.bar(
                        df,
                        x=cat_col1,
                        y=num_col,
                        color=cat_col2,
                        barmode="group",
                        title=f"{num_col} by {cat_col1} and {cat_col2}",
                        color_discrete_sequence=[
                            domain_config["primary_color"],
                            domain_config["secondary_color"],
                        ],
                    )
                    charts.append(
                        {
                            "type": "bar",
                            "title": f"{num_col} by {cat_col1} and {cat_col2}",
                            "spec": fig.to_json(),
                            "description": f"Grouped bar chart showing {num_col} values across {cat_col1} and {cat_col2}",
                        }
                    )
            except Exception as e:
                logging.warning(f"Failed to create grouped bar chart: {e}")

        return charts

    @staticmethod
    def _generate_line_charts(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate line charts."""
        charts = []
        numeric_cols = analysis["numeric_cols"]
        datetime_cols = analysis["datetime_cols"]
        categorical_cols = analysis["categorical_cols"]

        # Time series line charts
        if datetime_cols and numeric_cols:
            datetime_col = datetime_cols[0]
            for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                try:
                    # Aggregate by time if too many data points
                    if len(df) > 1000:
                        df_agg = (
                            df.set_index(datetime_col)
                            .resample("D")[num_col]
                            .mean()
                            .reset_index()
                        )
                    else:
                        df_agg = df[[datetime_col, num_col]].copy()

                    fig = px.line(
                        df_agg,
                        x=datetime_col,
                        y=num_col,
                        title=f"{num_col} over Time",
                        color_discrete_sequence=[domain_config["primary_color"]],
                    )
                    charts.append(
                        {
                            "type": "line",
                            "title": f"{num_col} over Time",
                            "spec": fig.to_json(),
                            "description": f"Line chart showing {num_col} trends over time",
                        }
                    )
                except Exception as e:
                    logging.warning(
                        f"Failed to create line chart for {num_col} over time: {e}"
                    )

        # Multi-line charts with categories
        if datetime_cols and numeric_cols and categorical_cols:
            datetime_col = datetime_cols[0]
            num_col = numeric_cols[0]
            cat_col = categorical_cols[0]

            if df[cat_col].nunique() <= 10:  # Reasonable number of categories
                try:
                    fig = px.line(
                        df,
                        x=datetime_col,
                        y=num_col,
                        color=cat_col,
                        title=f"{num_col} over Time by {cat_col}",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                    )
                    charts.append(
                        {
                            "type": "line",
                            "title": f"{num_col} over Time by {cat_col}",
                            "spec": fig.to_json(),
                            "description": f"Multi-line chart showing {num_col} trends over time by {cat_col}",
                        }
                    )
                except Exception as e:
                    logging.warning(f"Failed to create multi-line chart: {e}")

        return charts

    @staticmethod
    def _generate_scatter_plots(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate scatter plots."""
        charts = []
        numeric_cols = analysis["numeric_cols"]
        categorical_cols = analysis["categorical_cols"]

        if len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]

            # Basic scatter plot
            try:
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}",
                    color_discrete_sequence=[domain_config["primary_color"]],
                )
                charts.append(
                    {
                        "type": "scatter",
                        "title": f"{y_col} vs {x_col}",
                        "spec": fig.to_json(),
                        "description": f"Scatter plot showing relationship between {y_col} and {x_col}",
                    }
                )
            except Exception as e:
                logging.warning(
                    f"Failed to create scatter plot for {x_col} and {y_col}: {e}"
                )

            # Colored by category if available
            if categorical_cols and df[categorical_cols[0]].nunique() <= 10:
                try:
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        color=categorical_cols[0],
                        title=f"{y_col} vs {x_col} by {categorical_cols[0]}",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                    )
                    charts.append(
                        {
                            "type": "scatter",
                            "title": f"{y_col} vs {x_col} by {categorical_cols[0]}",
                            "spec": fig.to_json(),
                            "description": f"Scatter plot showing relationship between {y_col} and {x_col} colored by {categorical_cols[0]}",
                        }
                    )
                except Exception as e:
                    logging.warning(f"Failed to create colored scatter plot: {e}")

        return charts

    @staticmethod
    def _generate_histograms(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate histograms."""
        charts = []
        numeric_cols = analysis["numeric_cols"]

        for num_col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            try:
                fig = px.histogram(
                    df,
                    x=num_col,
                    title=f"Distribution of {num_col}",
                    color_discrete_sequence=[domain_config["primary_color"]],
                )
                charts.append(
                    {
                        "type": "histogram",
                        "title": f"Distribution of {num_col}",
                        "spec": fig.to_json(),
                        "description": f"Histogram showing distribution of {num_col} values",
                    }
                )
            except Exception as e:
                logging.warning(f"Failed to create histogram for {num_col}: {e}")

        return charts

    @staticmethod
    def _generate_box_plots(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate box plots."""
        charts = []
        numeric_cols = analysis["numeric_cols"]
        categorical_cols = analysis["categorical_cols"]

        # Box plots for numeric variables
        for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            try:
                fig = px.box(
                    df,
                    y=num_col,
                    title=f"Box Plot of {num_col}",
                    color_discrete_sequence=[domain_config["primary_color"]],
                )
                charts.append(
                    {
                        "type": "box",
                        "title": f"Box Plot of {num_col}",
                        "spec": fig.to_json(),
                        "description": f"Box plot showing distribution and outliers of {num_col}",
                    }
                )
            except Exception as e:
                logging.warning(f"Failed to create box plot for {num_col}: {e}")

        # Box plots by category
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]

            if df[cat_col].nunique() <= 10:  # Reasonable number of categories
                try:
                    fig = px.box(
                        df,
                        x=cat_col,
                        y=num_col,
                        title=f"{num_col} by {cat_col}",
                        color_discrete_sequence=[domain_config["primary_color"]],
                    )
                    charts.append(
                        {
                            "type": "box",
                            "title": f"{num_col} by {cat_col}",
                            "spec": fig.to_json(),
                            "description": f"Box plot showing distribution of {num_col} across {cat_col} categories",
                        }
                    )
                except Exception as e:
                    logging.warning(
                        f"Failed to create box plot for {num_col} by {cat_col}: {e}"
                    )

        return charts

    @staticmethod
    def _generate_violin_plots(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate violin plots."""
        charts = []
        numeric_cols = analysis["numeric_cols"]
        categorical_cols = analysis["categorical_cols"]

        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]

            if df[cat_col].nunique() <= 10:  # Reasonable number of categories
                try:
                    fig = px.violin(
                        df,
                        x=cat_col,
                        y=num_col,
                        title=f"Distribution of {num_col} by {cat_col}",
                        color_discrete_sequence=[domain_config["primary_color"]],
                    )
                    charts.append(
                        {
                            "type": "violin",
                            "title": f"Distribution of {num_col} by {cat_col}",
                            "spec": fig.to_json(),
                            "description": f"Violin plot showing distribution of {num_col} across {cat_col} categories",
                        }
                    )
                except Exception as e:
                    logging.warning(
                        f"Failed to create violin plot for {num_col} by {cat_col}: {e}"
                    )

        return charts

    @staticmethod
    def _generate_pie_charts(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate pie charts."""
        charts = []
        categorical_cols = analysis["categorical_cols"]
        numeric_cols = analysis["numeric_cols"]

        for cat_col in categorical_cols:
            if df[cat_col].nunique() <= 10:  # Good for pie charts
                try:
                    # Use value counts for the pie chart
                    value_counts = df[cat_col].value_counts()

                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Distribution of {cat_col}",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                    )
                    charts.append(
                        {
                            "type": "pie",
                            "title": f"Distribution of {cat_col}",
                            "spec": fig.to_json(),
                            "description": f"Pie chart showing distribution of {cat_col} categories",
                        }
                    )
                except Exception as e:
                    logging.warning(f"Failed to create pie chart for {cat_col}: {e}")

        # Pie chart with numeric aggregation
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]

            if df[cat_col].nunique() <= 10:
                try:
                    fig = px.pie(
                        df,
                        names=cat_col,
                        values=num_col,
                        title=f"Proportion of {num_col} by {cat_col}",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                    )
                    charts.append(
                        {
                            "type": "pie",
                            "title": f"Proportion of {num_col} by {cat_col}",
                            "spec": fig.to_json(),
                            "description": f"Pie chart showing proportion of {num_col} across {cat_col} categories",
                        }
                    )
                except Exception as e:
                    logging.warning(
                        f"Failed to create pie chart for {num_col} by {cat_col}: {e}"
                    )

        return charts

    @staticmethod
    def _generate_heatmaps(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate heatmaps."""
        charts = []
        numeric_cols = analysis["numeric_cols"]
        categorical_cols = analysis["categorical_cols"]

        # Correlation heatmap
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()

                fig = px.imshow(
                    corr_matrix,
                    title="Correlation Matrix",
                    color_continuous_scale=px.colors.diverging.RdBu_r,
                    aspect="auto",
                )
                charts.append(
                    {
                        "type": "heatmap",
                        "title": "Correlation Matrix",
                        "spec": fig.to_json(),
                        "description": "Heatmap showing correlations between numeric variables",
                    }
                )
            except Exception as e:
                logging.warning(f"Failed to create correlation heatmap: {e}")

        # Categorical heatmap
        if len(categorical_cols) >= 2:
            cat_col1, cat_col2 = categorical_cols[0], categorical_cols[1]

            if df[cat_col1].nunique() <= 20 and df[cat_col2].nunique() <= 20:
                try:
                    cross_tab = pd.crosstab(df[cat_col1], df[cat_col2])

                    fig = px.imshow(
                        cross_tab,
                        title=f"Relationship between {cat_col1} and {cat_col2}",
                        color_continuous_scale=px.colors.sequential.Blues,
                        aspect="auto",
                    )
                    charts.append(
                        {
                            "type": "heatmap",
                            "title": f"Relationship between {cat_col1} and {cat_col2}",
                            "spec": fig.to_json(),
                            "description": f"Heatmap showing relationship between {cat_col1} and {cat_col2}",
                        }
                    )
                except Exception as e:
                    logging.warning(f"Failed to create categorical heatmap: {e}")

        return charts

    @staticmethod
    def _generate_correlation_matrix(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate correlation matrix."""
        charts = []
        numeric_cols = analysis["numeric_cols"]

        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()

                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                    )
                )

                fig.update_layout(title="Correlation Matrix")

                charts.append(
                    {
                        "type": "correlation",
                        "title": "Correlation Matrix",
                        "spec": fig.to_json(),
                        "description": "Correlation matrix showing relationships between numeric variables",
                    }
                )
            except Exception as e:
                logging.warning(f"Failed to create correlation matrix: {e}")

        return charts

    @staticmethod
    def _generate_treemaps(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate treemaps."""
        charts = []
        categorical_cols = analysis["categorical_cols"]
        numeric_cols = analysis["numeric_cols"]

        if len(categorical_cols) >= 2 and numeric_cols:
            path = categorical_cols[
                :2
            ]  # Use first two categorical columns for hierarchy
            value_col = numeric_cols[0]

            try:
                fig = px.treemap(
                    df,
                    path=path,
                    values=value_col,
                    title=f"Treemap of {value_col} by {', '.join(path)}",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                charts.append(
                    {
                        "type": "treemap",
                        "title": f"Treemap of {value_col} by {', '.join(path)}",
                        "spec": fig.to_json(),
                        "description": f"Treemap showing hierarchical relationship of {value_col} across {', '.join(path)}",
                    }
                )
            except Exception as e:
                logging.warning(f"Failed to create treemap: {e}")

        return charts

    @staticmethod
    def _generate_sunburst_charts(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate sunburst charts."""
        charts = []
        categorical_cols = analysis["categorical_cols"]
        numeric_cols = analysis["numeric_cols"]

        if len(categorical_cols) >= 2 and numeric_cols:
            path = categorical_cols[
                :2
            ]  # Use first two categorical columns for hierarchy
            value_col = numeric_cols[0]

            try:
                fig = px.sunburst(
                    df,
                    path=path,
                    values=value_col,
                    title=f"Sunburst Chart of {value_col} by {', '.join(path)}",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                charts.append(
                    {
                        "type": "sunburst",
                        "title": f"Sunburst Chart of {value_col} by {', '.join(path)}",
                        "spec": fig.to_json(),
                        "description": f"Sunburst chart showing hierarchical relationship of {value_col} across {', '.join(path)}",
                    }
                )
            except Exception as e:
                logging.warning(f"Failed to create sunburst chart: {e}")

        return charts

    @staticmethod
    def _generate_funnel_charts(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate funnel charts."""
        charts = []
        categorical_cols = analysis["categorical_cols"]
        numeric_cols = analysis["numeric_cols"]

        if categorical_cols and numeric_cols:
            stage_col = categorical_cols[0]
            value_col = numeric_cols[0]

            if df[stage_col].nunique() <= 10:  # Reasonable number of stages
                try:
                    # Aggregate values by stage
                    funnel_data = df.groupby(stage_col)[value_col].sum().reset_index()

                    fig = px.funnel(
                        funnel_data,
                        x=value_col,
                        y=stage_col,
                        title=f"Funnel Chart of {value_col} by {stage_col}",
                        color_discrete_sequence=[domain_config["primary_color"]],
                    )
                    charts.append(
                        {
                            "type": "funnel",
                            "title": f"Funnel Chart of {value_col} by {stage_col}",
                            "spec": fig.to_json(),
                            "description": f"Funnel chart showing {value_col} values across {stage_col} stages",
                        }
                    )
                except Exception as e:
                    logging.warning(f"Failed to create funnel chart: {e}")

        return charts

    @staticmethod
    def _generate_candlestick_charts(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate candlestick charts."""
        charts = []
        datetime_cols = analysis["datetime_cols"]
        numeric_cols = analysis["numeric_cols"]

        if datetime_cols and len(numeric_cols) >= 4:
            datetime_col = datetime_cols[0]

            # Look for OHLC-like columns (Open, High, Low, Close)
            ohlc_patterns = {
                "open": ["open", "opening", "start"],
                "high": ["high", "max", "maximum"],
                "low": ["low", "min", "minimum"],
                "close": ["close", "closing", "end"],
            }

            ohlc_cols = {}
            for ohlc_key, patterns in ohlc_patterns.items():
                for col in numeric_cols:
                    if any(pattern in col.lower() for pattern in patterns):
                        ohlc_cols[ohlc_key] = col
                        break

            # If we found all OHLC columns, create candlestick chart
            if len(ohlc_cols) == 4:
                try:
                    fig = go.Figure(
                        data=[
                            go.Candlestick(
                                x=df[datetime_col],
                                open=df[ohlc_cols["open"]],
                                high=df[ohlc_cols["high"]],
                                low=df[ohlc_cols["low"]],
                                close=df[ohlc_cols["close"]],
                            )
                        ]
                    )

                    fig.update_layout(
                        title="Candlestick Chart",
                        xaxis_title="Date",
                        yaxis_title="Price",
                    )

                    charts.append(
                        {
                            "type": "candlestick",
                            "title": "Candlestick Chart",
                            "spec": fig.to_json(),
                            "description": "Candlestick chart showing price movements over time",
                        }
                    )
                except Exception as e:
                    logging.warning(f"Failed to create candlestick chart: {e}")

        return charts

    @staticmethod
    def _generate_area_charts(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate area charts."""
        charts = []
        datetime_cols = analysis["datetime_cols"]
        numeric_cols = analysis["numeric_cols"]

        if datetime_cols and numeric_cols:
            datetime_col = datetime_cols[0]
            num_col = numeric_cols[0]

            try:
                fig = px.area(
                    df,
                    x=datetime_col,
                    y=num_col,
                    title=f"{num_col} over Time (Area Chart)",
                    color_discrete_sequence=[domain_config["primary_color"]],
                )
                charts.append(
                    {
                        "type": "area",
                        "title": f"{num_col} over Time (Area Chart)",
                        "spec": fig.to_json(),
                        "description": f"Area chart showing {num_col} values over time",
                    }
                )
            except Exception as e:
                logging.warning(f"Failed to create area chart for {num_col}: {e}")

        return charts

    @staticmethod
    def _generate_bubble_charts(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate bubble charts."""
        charts = []
        numeric_cols = analysis["numeric_cols"]
        categorical_cols = analysis["categorical_cols"]

        if len(numeric_cols) >= 3:
            x_col, y_col, size_col = numeric_cols[0], numeric_cols[1], numeric_cols[2]

            try:
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    size=size_col,
                    title=f"Bubble Chart: {y_col} vs {x_col} (Size: {size_col})",
                    color_discrete_sequence=[domain_config["primary_color"]],
                )
                charts.append(
                    {
                        "type": "bubble",
                        "title": f"Bubble Chart: {y_col} vs {x_col}",
                        "spec": fig.to_json(),
                        "description": f"Bubble chart showing relationship between {y_col} and {x_col} with size representing {size_col}",
                    }
                )
            except Exception as e:
                logging.warning(f"Failed to create bubble chart: {e}")

            # Colored by category if available
            if categorical_cols and df[categorical_cols[0]].nunique() <= 10:
                try:
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        size=size_col,
                        color=categorical_cols[0],
                        title=f"Bubble Chart: {y_col} vs {x_col} by {categorical_cols[0]}",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                    )
                    charts.append(
                        {
                            "type": "bubble",
                            "title": f"Bubble Chart: {y_col} vs {x_col} by {categorical_cols[0]}",
                            "spec": fig.to_json(),
                            "description": f"Bubble chart showing relationship between {y_col} and {x_col} colored by {categorical_cols[0]} with size representing {size_col}",
                        }
                    )
                except Exception as e:
                    logging.warning(f"Failed to create colored bubble chart: {e}")

        return charts

    @staticmethod
    def _generate_radar_charts(
        df: pd.DataFrame, analysis: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate radar charts."""
        charts = []
        numeric_cols = analysis["numeric_cols"]
        categorical_cols = analysis["categorical_cols"]

        if categorical_cols and len(numeric_cols) >= 3:
            cat_col = categorical_cols[0]

            if df[cat_col].nunique() <= 5:  # Limit number of categories for readability
                try:
                    # Normalize numeric values for radar chart
                    numeric_subset = numeric_cols[:5]  # Use first 5 numeric columns
                    df_normalized = df[numeric_subset + [cat_col]].copy()

                    for col in numeric_subset:
                        df_normalized[col] = (
                            df_normalized[col] - df_normalized[col].min()
                        ) / (df_normalized[col].max() - df_normalized[col].min())

                    # Create radar chart for each category
                    for category in df_normalized[cat_col].unique():
                        category_data = df_normalized[
                            df_normalized[cat_col] == category
                        ]
                        mean_values = category_data[numeric_subset].mean()

                        fig = go.Figure()

                        fig.add_trace(
                            go.Scatterpolar(
                                r=mean_values.values,
                                theta=numeric_subset,
                                fill="toself",
                                name=category,
                            )
                        )

                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            title=f"Radar Chart: {cat_col} = {category}",
                            showlegend=True,
                        )

                        charts.append(
                            {
                                "type": "radar",
                                "title": f"Radar Chart: {cat_col} = {category}",
                                "spec": fig.to_json(),
                                "description": f"Radar chart showing normalized values of numeric variables for {cat_col} = {category}",
                            }
                        )
                except Exception as e:
                    logging.warning(f"Failed to create radar chart: {e}")

        return charts


# import pandas as pd
# import logging
# from typing import List, Dict, Any

# # --- NEW: Import Plotly ---
# import plotly.express as px

# # --- Basic Logging Setup ---
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )


# class VisualizationService:
#     """
#     A service class to handle the logic of generating interactive chart specifications.
#     """

#     @staticmethod
#     def generate_visualizations(
#         table_data: List[Dict[str, Any]], domain: str = "generic"
#     ) -> List[str]:
#         """
#         Takes table data, generates interactive charts using Plotly,
#         and returns a list of JSON strings representing those charts.

#         Args:
#             table_data: A list of dictionaries, where each dictionary represents a row.
#             domain: The industry domain (not used in this basic example, but kept for structure).

#         Returns:
#             A list of strings, where each string is the JSON specification for a chart.
#         """
#         if not table_data:
#             logging.warning("Visualization request received with no data.")
#             return []

#         try:
#             # 1. Convert data into a pandas DataFrame (this part stays the same)
#             df = pd.DataFrame(table_data)
#             logging.info(
#                 f"Successfully created DataFrame with {df.shape[0]} rows and {df.shape[1]} columns."
#             )

#             # --- THIS IS THE NEW LOGIC ---

#             # 2. Create a list to hold our chart JSON
#             chart_specs = []

#             # 3. Identify column types for basic chart generation
#             numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
#             categorical_cols = df.select_dtypes(
#                 include=["object", "category"]
#             ).columns.tolist()

#             # 4. Generate some example charts based on available data types

#             # Chart Example 1: A bar chart if we have at least one categorical and one numeric column
#             if categorical_cols and numeric_cols:
#                 logging.info("Generating a bar chart.")
#                 # We'll use the first column of each type for simplicity
#                 fig_bar = px.bar(
#                     df,
#                     x=categorical_cols[0],
#                     y=numeric_cols[0],
#                     title=f"{numeric_cols[0]} by {categorical_cols[0]}",
#                 )
#                 chart_specs.append(fig_bar.to_json())

#             # Chart Example 2: A scatter plot if we have at least two numeric columns
#             if len(numeric_cols) >= 2:
#                 logging.info("Generating a scatter plot.")
#                 fig_scatter = px.scatter(
#                     df,
#                     x=numeric_cols[0],
#                     y=numeric_cols[1],
#                     title=f"{numeric_cols[1]} vs. {numeric_cols[0]}",
#                 )
#                 chart_specs.append(fig_scatter.to_json())

#             # You can add more rules here for other chart types (histograms, line charts, etc.)

#             logging.info(f"Generated {len(chart_specs)} chart specifications.")

#             # 5. Return the list of JSON strings
#             return chart_specs

#         except Exception as e:
#             logging.error(f"An error occurred during visualization: {e}", exc_info=True)
#             return []
