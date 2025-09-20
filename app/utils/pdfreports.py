# C:\Users\Vernon\Desktop\datacura\backend\app\utils\pdfreports.py

from io import BytesIO
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import numpy as np

from fpdf import FPDF


class PDF(FPDF):
    """ Custom PDF class with a CURAREFINE header and a standard footer. """

    def header(self):
        # Logo - wrapped in a try-except block to prevent errors if the file is missing
        try:
            # Assuming the logo is stored in an 'assets' folder relative to the execution path
            self.image("app/assets/cura_icons.png", 10, 8, 15)
        except RuntimeError:
            # If the logo is not found, we just print a warning and continue without it
            print("Warning: Logo file 'app/assets/cura_icons.png' not found.")
            pass

        # Report Title
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "CURAREFINE DATA ANALYSIS REPORT", border=False, ln=1, align="C")
        # Line break
        self.ln(10)

    def footer(self):
        """ Adds a page number to the bottom of each page. """
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, title):
        """ Creates a formatted chapter title. """
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        """ Creates formatted chapter body text. """
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_dataframe(self, df):
        """ Renders a pandas DataFrame as a table in the PDF. """
        self.set_font("Arial", 'B', 8)
        # Header
        for col in df.columns:
            self.cell(35, 7, col, 1)
        self.ln()
        # Data
        self.set_font("Arial", '', 8)
        for index, row in df.iterrows():
            for item in row:
                self.cell(35, 7, str(item), 1)
            self.ln()
        self.ln()


def generate_comprehensive_report(
        current_data: list[dict],
        project_name: str
) -> BytesIO:
    """
    Generates a comprehensive PDF report from the current state of a dataset.

    Args:
        current_data: A list of dictionaries representing the rows of the current dataset.
        project_name: The name of the project/file.

    Returns:
        A BytesIO object containing the generated PDF.
    """
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Convert incoming data to DataFrame
    df = pd.DataFrame(current_data)
    # Exclude the '__id' column added by the frontend from analysis
    if '__id' in df.columns:
        df = df.drop(columns=['__id'])

    # --- Cover Page ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 20, "Comprehensive Data Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Project: {project_name}", ln=True, align="C")
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")

    # --- 1. Data Overview ---
    pdf.add_page()
    pdf.chapter_title("1. Data Overview")
    overview_text = (
        f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n"
        f"Data Types Found:\n{df.dtypes.to_string()}"
    )
    pdf.chapter_body(overview_text)

    # --- 2. Descriptive Statistics (for numeric columns) ---
    numeric_cols = df.select_dtypes(include=np.number)
    if not numeric_cols.empty:
        pdf.chapter_title("2. Descriptive Statistics")
        stats_df = numeric_cols.describe().round(2).reset_index()
        pdf.chapter_body("Summary of numerical columns:")
        pdf.add_dataframe(stats_df.rename(columns={'index': 'Statistic'}))

    # --- 3. Missing Value Analysis ---
    pdf.add_page()
    pdf.chapter_title("3. Missing Value Analysis")
    missing_values = df.isnull().sum()
    missing_df = missing_values[missing_values > 0].sort_values(ascending=False).reset_index()
    missing_df.columns = ["Column", "Missing Count"]

    if not missing_df.empty:
        missing_df["Missing %"] = (missing_df["Missing Count"] / df.shape[0] * 100).round(2)
        pdf.chapter_body("Columns with missing values:")
        pdf.add_dataframe(missing_df)

        # Missing values bar plot
        plt.figure(figsize=(8, 5))
        missing_df.set_index("Column")["Missing %"].plot(kind="bar", color="skyblue")
        plt.title("Percentage of Missing Values by Column")
        plt.ylabel("% Missing")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, delete_on_close=False) as tmpfile:
            plt.savefig(tmpfile.name, dpi=150)
            pdf.image(tmpfile.name, w=170)
        plt.close()
    else:
        pdf.chapter_body("No missing values were detected in the dataset. ✅")

    # --- 4. Outlier Visualization ---
    if not numeric_cols.empty:
        pdf.add_page()
        pdf.chapter_title("4. Outlier Visualization (Box Plots)")
        pdf.chapter_body(
            "Box plots are used to visualize the distribution of numeric data and identify potential outliers.")

        for col in numeric_cols.columns:
            plt.figure(figsize=(8, 4))
            plt.boxplot(numeric_cols[col].dropna(), vert=False)
            plt.title(f"Distribution of '{col}'")
            plt.tight_layout()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False, delete_on_close=False) as tmpfile:
                plt.savefig(tmpfile.name, dpi=120)
                pdf.image(tmpfile.name, w=160)
                pdf.ln(5)
            plt.close()

    # --- Return as BytesIO ---
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_bytes)