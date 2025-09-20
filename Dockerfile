# Use an official, slim Python runtime as the base image
FROM python:3.11-slim

# Set environment variables to avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV APP_HOME=/app
WORKDIR $APP_HOME

# Install system dependencies required by matplotlib, pandas, and pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the Cloud Run port
EXPOSE 8080

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
