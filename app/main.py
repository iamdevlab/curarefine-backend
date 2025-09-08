import asyncio
import os
from pathlib import Path

# Third-Party Libraries
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# --- Local Application Imports ---
# Initialize environment variables first
load_dotenv()

# Import all the API routers
from app.services.ai_service import router as ai_service_router
from app.api.ai_settings import router as ai_settings_router
from app.api.auth import router as auth_router
from app.api.clean import router as clean_router
from app.api.dashboard import router as dashboard_router
from app.api.export import router as export_router

# from app.deep_analysis.pipeline import router as pipeline_router
from app.api.upload import router as upload_router
from app.api.users import router as users_router
from app.api.visualize_action import router as visualize_action_router

# Import services and utilities
from app.services.postgres_client import init_db_async
from app.services.redis_client import init_redis
from app.utils.json_response import CustomJSONResponse as JSONResponse

# Create FastAPI app
app = FastAPI(
    title="Data Cleaning API",
    description="AI-powered data cleaning and processing service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=JSONResponse,
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

origins = [
    "https://2969fdaa.curarefine-frontend.pages.dev",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://localhost:5173",
]  # Allow all origins for Replit environment

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- MODIFIED: Include BOTH AI routers ---
app.include_router(upload_router, prefix="/api/upload")
app.include_router(clean_router, prefix="/api")
app.include_router(export_router, prefix="/api")
# This includes the endpoints like /ai/capabilities
app.include_router(ai_service_router, prefix="/api")
# This includes the new endpoints like /ai/providers and /ai/settings
app.include_router(ai_settings_router, prefix="/api")
# app.include_router(pipeline.router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(visualize_action_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(users_router, prefix="/api")


# --- (The rest of your original file is preserved below) ---


# Create necessary directories
def create_directories():
    """Create required directories on startup"""
    directories = ["uploads", "exports", "temp"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    create_directories()

    # Initialize Redis first (it's usually faster), s
    init_redis()

    # Initialize DB with timeout
    try:
        await asyncio.wait_for(init_db_async(), timeout=10.0)
    except asyncio.TimeoutError:
        print("[Postgres] Database initialization timed out - continuing without DB")
    except Exception as e:
        print(f"[Postgres] Database initialization failed: {e}")

    print("Data Cleaning API started...")


@app.get("/api/status")
async def status_check():
    """Endpoint for health and status checks"""
    return JSONResponse(
        {
            "status": "online",
            "message": "Data Cleaning API is running via automated CI/CD!",
            "routes": {
                "upload": {
                    "POST /upload/file": "Upload a new data file",
                    "POST /upload/url": "Upload data from a URL",
                },
                "clean": {
                    "GET /clean/preview/{file_id}": "Get a preview of the data",
                    "POST /clean/missing": "Handle missing values",
                    "POST /clean/duplicates": "Remove duplicate rows",
                    "POST /clean/text": "Clean text columns",
                    "POST /clean/outliers": "Handle outliers",
                    "POST /clean/convert-types": "Convert data types",
                    "POST /clean/standardize-columns": "Standardize column names",
                    "GET /clean/recommendations/{file_id}": "Get cleaning recommendations",
                    "DELETE /clean/session/{file_id}": "End cleaning session",
                },
                "export": {
                    "GET /export/status/{file_id}": "Get export status",
                    "GET /export/{file_id}": "Export cleaned data",
                    "POST /export/multiple": "Export in multiple formats",
                    "GET /export/report/{file_id}": "Export cleaning report",
                    "GET /export/formats": "Get supported formats",
                    "POST /export/preview": "Preview export data",
                },
                "ai": {
                    "POST /ai/analyze": "AI-powered data analysis",
                    "POST /ai/auto-clean": "Automatic data cleaning",
                    "POST /ai/insights": "Get AI insights",
                    "GET /ai/capabilities": "Get AI capabilities",
                    "POST /ai/validate-recommendation": "Validate cleaning recommendation",
                },
            },
        }
    )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "documentation": "/docs",
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "support": "Please check the logs or contact support",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
