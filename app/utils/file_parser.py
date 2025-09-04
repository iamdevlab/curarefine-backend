# app/utils/file_parser.py
import pandas as pd
import io, os


def read_file_content(content: bytes, filename: str) -> pd.DataFrame:
    """
    Reads file content into a Pandas DataFrame based on extension.
    Supports CSV, Excel, JSON, and TXT.
    """
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext in [".csv"]:
        return pd.read_csv(io.BytesIO(content))

    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(io.BytesIO(content))

    elif ext in [".json"]:
        return pd.read_json(io.BytesIO(content))

    elif ext in [".txt"]:
        # Treat txt as CSV with whitespace or tab separation
        try:
            return pd.read_csv(io.BytesIO(content), sep=None, engine="python")
        except Exception:
            return pd.DataFrame({"text": content.decode("utf-8").splitlines()})

    else:
        raise ValueError(f"Unsupported file extension: {ext}")
