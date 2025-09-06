import pandas as pd
import io

import docx
from pathlib import Path
import chardet
from typing import Tuple, Dict, Any, List


from app.models.schemas import FileType, UploadResponse


class FileProcessor:

    @staticmethod
    def detect_file_type(filename: str, content: bytes) -> FileType:
        """Detect file type from filename and content"""
        suffix = Path(filename).suffix.lower()

        if suffix in [".csv"]:
            return FileType.CSV
        elif suffix in [".xlsx", ".xls"]:
            return FileType.EXCEL
        elif suffix in [".pdf"]:
            return FileType.PDF
        elif suffix in [".docx", ".doc"]:
            return FileType.WORD
        elif suffix in [".txt"]:
            return FileType.TEXT
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    @staticmethod
    def detect_encoding(content: bytes) -> str:
        """Detect text encoding"""
        result = chardet.detect(content)
        return result["encoding"] or "utf-8"

    @staticmethod
    def process_csv(content: bytes) -> pd.DataFrame:
        """Process CSV file"""
        encoding = FileProcessor.detect_encoding(content)
        try:
            # Try comma separator first
            df = pd.read_csv(io.StringIO(content.decode(encoding)))
        except:
            # Try semicolon separator
            df = pd.read_csv(io.StringIO(content.decode(encoding)), sep=";")
        return df

    @staticmethod
    def process_excel(content: bytes) -> pd.DataFrame:
        """Process Excel file"""
        return pd.read_excel(io.BytesIO(content))

    @staticmethod
    def process_pdf(content: bytes) -> pd.DataFrame:
        """Process PDF file (extract tables)"""
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(content)) as pdf:
                all_tables = []
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            all_tables.extend(table)

                if all_tables:
                    df = pd.DataFrame(all_tables[1:], columns=all_tables[0])
                    return df
                else:
                    raise ValueError("No tables found in PDF")
        except ImportError:
            raise ValueError("pdfplumber not installed for PDF processing")

    @staticmethod
    def process_word(content: bytes) -> pd.DataFrame:
        """Process Word document (extract tables)"""
        doc = docx.Document(io.BytesIO(content))
        tables_data = []

        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)

            if table_data:
                tables_data.extend(table_data)

        if tables_data:
            df = pd.DataFrame(tables_data[1:], columns=tables_data[0])
            return df
        else:
            raise ValueError("No tables found in Word document")

    @staticmethod
    def process_text(content: bytes) -> pd.DataFrame:
        """Process text file (assume it's structured data)"""
        encoding = FileProcessor.detect_encoding(content)
        text = content.decode(encoding)

        # Try to detect delimiter
        lines = text.strip().split("\n")
        if not lines:
            raise ValueError("Empty text file")

        # Try different delimiters
        for delimiter in ["\t", ",", ";", "|"]:
            if delimiter in lines[0]:
                data = [line.split(delimiter) for line in lines]
                df = pd.DataFrame(data[1:], columns=data[0])
                return df

        # If no delimiter found, treat as single column
        df = pd.DataFrame(lines, columns=["text"])
        return df

    @classmethod
    def process_file(
        cls, filename: str, content: bytes
    ) -> Tuple[pd.DataFrame, FileType]:
        """Main file processing method"""
        file_type = cls.detect_file_type(filename, content)

        if file_type == FileType.CSV:
            df = cls.process_csv(content)
        elif file_type == FileType.EXCEL:
            df = cls.process_excel(content)
        elif file_type == FileType.PDF:
            df = cls.process_pdf(content)
        elif file_type == FileType.WORD:
            df = cls.process_word(content)
        elif file_type == FileType.TEXT:
            df = cls.process_text(content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return df, file_type
