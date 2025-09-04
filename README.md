# Data Cleaning API

A comprehensive, AI-powered data cleaning and processing service built with FastAPI. This API provides automated data quality analysis, intelligent cleaning suggestions, and multiple export formats.

## 🚀 Features

- **File Upload**: Support for CSV, Excel (XLSX/XLS), and JSON formats
- **AI-Powered Analysis**: Intelligent pattern detection and cleaning recommendations
- **Data Quality Assessment**: Comprehensive quality scoring and issue detection
- **Automated Cleaning**: Multiple strategies for handling missing values, duplicates, outliers
- **Text Processing**: Advanced text cleaning and standardization
- **Data Type Conversion**: Smart type inference and conversion
- **Multiple Export Formats**: CSV, Excel, JSON, Parquet with compression
- **Cleaning Reports**: Detailed reports of all cleaning operations
- **Interactive API**: Full OpenAPI/Swagger documentation

## 📦 Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd data-cleaning-api
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Core Endpoints

#### 🔄 File Upload
```
POST /upload/file              # Upload single file
POST /upload/multiple          # Upload multiple files
GET  /upload/info/{file_id}    # Get file information
GET  /upload/preview/{file_id} # Preview file data
DELETE /upload/{file_id}       # Delete uploaded file
```

#### 🧹 Data Cleaning
```
GET  /clean/analyze/{file_id}        # Analyze data quality
POST /clean/missing-values           # Handle missing values
POST /clean/duplicates              # Remove duplicates
POST /clean/text                    # Clean text columns
POST /clean/outliers                # Handle outliers
POST /clean/convert-types           # Convert data types
POST /clean/standardize-columns     # Standardize column names
POST /clean/pipeline                # Run cleaning pipeline
GET  /clean/report/{file_id}        # Get cleaning report
POST /clean/reset/{file_id}         # Reset to original data
GET  /clean/preview/{file_id}       # Preview cleaned data
```

#### 📤 Data Export
```
POST /export/download               # Export cleaned data
POST /export/multiple               # Export in multiple formats
GET  /export/report/{file_id}       # Export cleaning report
GET  /export/formats                # Get supported formats
POST /export/preview                # Preview export data
GET  /export/status/{file_id}       # Get export status
```

#### 🤖 AI Assistance
```
POST /ai/analyze                    # AI-powered data analysis
POST /ai/auto-clean                 # Automatic data cleaning
POST /ai/insights                   # Get AI insights
GET  /ai/capabilities               # Get AI capabilities
POST /ai/validate-recommendation    # Validate cleaning recommendation
```

## 🎯 Usage Examples

### 1. Upload a File
```python
import requests

# Upload CSV file
with open('data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload/file',
        files={'file': f}
    )
    
file_info = response.json()
file_id = file_info['file_info']['file_id']
```

### 2. Analyze Data Quality
```python
# Get AI-powered analysis
response = requests.get(f'http://localhost:8000/clean/analyze/{file_id}')
analysis = response.json()

print(f"Data Quality Issues: {analysis['detected_issues']}")
print(f"Recommendations: {analysis['recommendations']}")
```

### 3. Auto-Clean Data
```python
# Automatic cleaning with moderate aggressiveness
payload = {
    "file_id": file_id,
    "aggressive_level": "moderate",
    "preserve_columns": ["id", "timestamp"]
}

response = requests.post(
    'http://localhost:8000/ai/auto-clean',
    json=payload
)

cleaning_result = response.json()
print(f"Applied {cleaning_result['applied_steps']} cleaning steps")
```

### 4. Manual Cleaning Operations
```python
# Handle missing values
missing_payload = {
    "file_id": file_id,
    "strategy": "fill_median",
    "columns": ["age", "income"]
}

response = requests.post(
    'http://localhost:8000/clean/missing-values',
    json=missing_payload
)

# Remove duplicates
duplicate_payload = {
    "file_id": file_id,
    "keep": "first"
}

response = requests.post(
    'http://localhost:8000/clean/duplicates',
    json=duplicate_payload
)
```

### 5. Export Cleaned Data
```python
# Export as Excel with cleaning report
export_payload = {
    "file_id": file_id,
    "format": "xlsx",
    "include_report": True,
    "filename": "cleaned_data"
}

response = requests.post(
    'http://localhost:8000/export/download',
    json=export_payload
)

# Save the exported file
with open('cleaned_data.xlsx', 'wb') as f:
    f.write(response.content)
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in your project root:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True

# Upload Configuration
MAX_FILE_SIZE=50MB
UPLOAD_DIR=uploads
SUPPORTED_FORMATS=csv,xlsx,xls,json

# Cleaning Configuration
DEFAULT_AGGRESSIVE_LEVEL=moderate
MAX_CLEANING_SESSIONS=100

# Export Configuration
EXPORT_DIR=exports
TEMP_DIR=temp
```

### Advanced Configuration
```python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    upload_dir: str = "uploads"
    supported_formats: list = ["csv", "xlsx", "xls", "json"]
    max_cleaning_sessions: int = 100
    
    class Config:
        env_file = ".env"

settings = Settings()
```

## 🧪 Data Cleaning Capabilities

### Missing Value Strategies
- **drop_rows**: Remove rows with missing values
- **drop_columns**: Remove columns with missing values
- **fill_mean**: Fill with column mean (numeric)
- **fill_median**: Fill with column median (numeric)
- **fill_mode**: Fill with most frequent value
- **fill_value**: Fill with custom value
- **forward_fill**: Forward fill missing values
- **backward_fill**: Backward fill missing values

### Text Cleaning Operations
- **strip**: Remove leading/trailing whitespace
- **lower/upper**: Convert case
- **remove_special_chars**: Remove special characters
- **remove_numbers**: Remove numeric characters
- **standardize_whitespace**: Normalize whitespace

### Outlier Detection Methods
- **IQR Method**: Interquartile Range (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
- **Z-Score Method**: Standard deviations from mean (±3σ)

### Data Type Conversions
- **datetime**: Convert to datetime objects
- **numeric**: Convert to numeric types
- **category**: Convert to categorical
- **string**: Convert to string type

## 🤖 AI Features

### Pattern Detection
The AI service can automatically detect:
- **Email addresses**: RFC-compliant email patterns
- **Phone numbers**: Various international formats
- **Dates**: Multiple date/time formats
- **Numeric data**: Numbers stored as text
- **Categorical data**: Low-cardinality string columns
- **Unique identifiers**: ID columns

### Quality Assessment
- **Data completeness**: Missing value analysis
- **Data consistency**: Duplicate detection
- **Data validity**: Format validation
- **Data accuracy**: Outlier detection
- **Overall quality score**: 0-100 composite score

### Auto-Cleaning Levels
- **Conservative**: Only high-priority, safe operations
- **Moderate**: High and medium priority operations
- **Aggressive**: All recommended operations

## 📊 Export Formats

### Supported Output Formats
- **CSV**: Comma-separated values
- **XLSX**: Excel spreadsheet with multiple sheets
- **JSON**: JavaScript Object Notation
- **Parquet**: Columnar storage format

### Export Options
- **Single format**: Export in one format
- **Multiple formats**: ZIP file with multiple formats
- **Include original**: Include original data for comparison
- **Include report**: Add cleaning report to export

## 🚨 Error Handling

The API provides comprehensive error handling:

```json
{
  "status": "error",
  "error_code": "FILE_NOT_FOUND",
  "message": "File with ID 'abc123' not found",
  "details": {
    "file_id": "abc123",
    "suggestion": "Check if the file was uploaded successfully"
  }
}
```

### Common Error Codes
- `FILE_NOT_FOUND`: File doesn't exist
- `UNSUPPORTED_FORMAT`: File format not supported
- `FILE_TOO_LARGE`: File exceeds size limit
- `INVALID_PARAMETERS`: Invalid request parameters
- `PROCESSING_ERROR`: Error during data processing

## 🔒 Security Considerations

### File Upload Security
- File size limits
- Format validation
- Virus scanning (recommended for production)
- Temporary file cleanup

### Data Privacy
- Files are temporarily stored
- No data persistence beyond session
- Memory cleanup after processing
- GDPR compliance considerations

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

### Example Test
```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_upload_file():
    with open("test_data.csv", "rb") as f:
        response = client.post(
            "/upload/file",
            files={"file": ("test_data.csv", f, "text/csv")}
        )
    assert response.status_code == 200
    assert "file_id" in response.json()["file_info"]
```

## 📈 Performance

### Optimization Tips
- Use Parquet format for large datasets
- Enable compression for exports
- Monitor memory usage for large files
- Use streaming for large file downloads

### Limits
- **File Size**: 50MB default (configurable)
- **Concurrent Sessions**: 100 default (configurable)
- **Memory Usage**: Monitored and logged

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run in development mode
uvicorn main:app --reload --log-level debug

# Run tests
pytest --cov=. --cov-report=html
```

## 📝 API Response Examples

### Successful Upload Response
```json
{
  "status": "success",
  "message": "File uploaded successfully",
  "file_info": {
    "file_id": "uuid-here",
    "filename": "data.csv",
    "size": 1024,
    "rows": 100,
    "columns": 5,
    "column_names": ["id", "name", "age", "city", "salary"],
    "data_types": {
      "id": "int64",
      "name": "object",
      "age": "float64"
    }
  },
  "quality_summary": {
    "missing_values": {"age": 5, "salary": 2},
    "duplicate_rows": 3,
    "total_missing": 7
  }
}
```

### AI Analysis Response
```json
{
  "file_id": "uuid-here",
  "analysis_type": "suggest_cleaning",
  "recommendations": [
    {
      "type": "missing_values",
      "column": "age",
      "strategy": "fill_median",
      "reason": "Numeric column, median is robust to outliers",
      "priority": "high",
      "parameters": {
        "columns": ["age"],
        "strategy": "fill_median"
      }
    }
  ],
  "total_recommendations": 4
}
```

## 🐛 Troubleshooting

### Common Issues

**File Upload Fails**
- Check file size limits
- Verify file format is supported
- Ensure file is not corrupted

**Memory Errors**
- Reduce file size
- Check available system memory
- Use streaming for large files

**Processing Timeout**
- Large files may take time
- Consider breaking into smaller chunks
- Monitor server resources

## 📞 Support

- **Documentation**: `/docs` endpoint
- **Health Check**: `/health` endpoint
- **API Info**: `/api/info` endpoint

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using FastAPI, Pandas, and AI-powered data processing**