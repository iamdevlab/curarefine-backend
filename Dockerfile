# Use an official, slim Python runtime as the base image
FROM python:3.11-slim

# Set the working directory inside the container
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Tell Cloud Run what command to run to start your app
# The port must be set to 8080 for Cloud Run's default
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]