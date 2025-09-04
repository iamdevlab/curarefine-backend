# app/services/visualizer.py

import pandas as pd
import logging
from typing import List, Dict, Any

# --- NEW: Import Plotly ---
import plotly.express as px

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class VisualizationService:
    """
    A service class to handle the logic of generating interactive chart specifications.
    """

    @staticmethod
    def generate_visualizations(
        table_data: List[Dict[str, Any]], domain: str = "generic"
    ) -> List[str]:
        """
        Takes table data, generates interactive charts using Plotly,
        and returns a list of JSON strings representing those charts.

        Args:
            table_data: A list of dictionaries, where each dictionary represents a row.
            domain: The industry domain (not used in this basic example, but kept for structure).

        Returns:
            A list of strings, where each string is the JSON specification for a chart.
        """
        if not table_data:
            logging.warning("Visualization request received with no data.")
            return []

        try:
            # 1. Convert data into a pandas DataFrame (this part stays the same)
            df = pd.DataFrame(table_data)
            logging.info(
                f"Successfully created DataFrame with {df.shape[0]} rows and {df.shape[1]} columns."
            )

            # --- THIS IS THE NEW LOGIC ---

            # 2. Create a list to hold our chart JSON
            chart_specs = []

            # 3. Identify column types for basic chart generation
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # 4. Generate some example charts based on available data types

            # Chart Example 1: A bar chart if we have at least one categorical and one numeric column
            if categorical_cols and numeric_cols:
                logging.info("Generating a bar chart.")
                # We'll use the first column of each type for simplicity
                fig_bar = px.bar(
                    df,
                    x=categorical_cols[0],
                    y=numeric_cols[0],
                    title=f"{numeric_cols[0]} by {categorical_cols[0]}",
                )
                chart_specs.append(fig_bar.to_json())

            # Chart Example 2: A scatter plot if we have at least two numeric columns
            if len(numeric_cols) >= 2:
                logging.info("Generating a scatter plot.")
                fig_scatter = px.scatter(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title=f"{numeric_cols[1]} vs. {numeric_cols[0]}",
                )
                chart_specs.append(fig_scatter.to_json())

            # You can add more rules here for other chart types (histograms, line charts, etc.)

            logging.info(f"Generated {len(chart_specs)} chart specifications.")

            # 5. Return the list of JSON strings
            return chart_specs

        except Exception as e:
            logging.error(f"An error occurred during visualization: {e}", exc_info=True)
            return []
