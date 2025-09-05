# Overview

This is an AI-powered data cleaning and processing service built with FastAPI. The application provides comprehensive data quality analysis, automated cleaning recommendations, and intelligent data transformation capabilities. Users can upload datasets in multiple formats (CSV, Excel, JSON), receive AI-driven insights about data quality issues, apply various cleaning strategies, and export the cleaned data in their preferred format.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Backend Framework
The application is built on **FastAPI** as the core web framework, providing automatic API documentation, request validation, and high-performance async capabilities. The modular router structure separates concerns into distinct API endpoints for upload, cleaning, export, authentication, and AI services.

## Data Processing Architecture
The system employs a **stateful data processing pipeline** where uploaded files are stored locally in an `uploads/` directory and processed through a multi-stage cleaning workflow. The core `DataCleaner` class manages data transformations while maintaining detailed operation logs. An advanced `DatacuraPipeline` orchestrates complex cleaning operations including outlier detection using multiple algorithms (IQR, Z-score, Isolation Forest).

## Authentication & Authorization
**JWT-based authentication** is implemented with cookie storage for session management. User credentials are securely hashed using bcrypt, and protected routes use dependency injection for user verification. The system supports user registration, password changes, and role-based access control.

## Data Storage Strategy
The application uses a **dual-storage approach**:
- **PostgreSQL** for persistent data including user accounts, project metadata, AI provider settings, and cleaning operation history
- **Redis** for temporary session storage and caching of data processing states
- **Local file system** for uploaded files with UUID-based naming for security

## AI Integration Layer
The system features a **provider-agnostic AI service** that supports multiple LLM providers (Anthropic, Cohere, Google, Ollama) through a normalized request adapter pattern. Users can configure their preferred AI provider and API keys, with intelligent fallback mechanisms for processing recommendations.

## Visualization System
**Plotly-based interactive visualizations** are generated dynamically from cleaned datasets. The visualization service creates chart specifications as JSON that can be rendered in the frontend, supporting various chart types based on data characteristics.

## Security Architecture
**Multi-layer security** includes:
- Environment-based encryption keys for sensitive data
- API key encryption for AI provider credentials
- CORS middleware for cross-origin request handling
- Input validation and file type restrictions
- JWT token expiration and refresh mechanisms

# External Dependencies

## AI Services
- **Multiple LLM Providers**: Anthropic Claude, Cohere, Google AI, Ollama for generating data cleaning recommendations and insights
- **Provider-specific APIs**: Each AI service requires separate API keys and uses different request formats handled by the adapter pattern

## Database Systems
- **PostgreSQL**: Primary database for user data, project metadata, and AI provider configurations
- **Redis**: Session storage and caching layer for temporary data processing states

## Data Processing Libraries
- **Pandas**: Core data manipulation and analysis
- **NumPy**: Numerical computations and array operations
- **Scikit-learn**: Machine learning algorithms for outlier detection (Isolation Forest)

## Web Framework Dependencies
- **FastAPI**: Core web framework with automatic documentation
- **Uvicorn**: ASGI server for running the FastAPI application
- **Pydantic**: Data validation and settings management

## Authentication & Security
- **python-jose**: JWT token creation and validation
- **passlib**: Password hashing with bcrypt
- **cryptography**: Encryption services for sensitive data

## File Processing
- **python-multipart**: File upload handling
- **chardet**: Character encoding detection for text files

## Visualization
- **Plotly**: Interactive chart generation and JSON specification creation

## Development & Deployment
- **python-dotenv**: Environment variable management
- **CORS middleware**: Cross-origin request handling for web client integration