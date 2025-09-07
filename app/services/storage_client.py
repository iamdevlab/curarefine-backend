import os
from google.cloud import storage
from pathlib import Path
from typing import Optional

# Load from env (set these in config/.env)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
# GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", "")

# Initialize storage client once
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)


def upload_file_to_gcs(local_path: str, destination_blob: Optional[str] = None) -> str:
    """
    Uploads a local file to GCS and returns the GCS path.
    """
    if not destination_blob:
        destination_blob = Path(local_path).name  # fallback to filename

    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(local_path)

    return f"gs://{GCS_BUCKET_NAME}/{destination_blob}"


def download_file_from_gcs(blob_name: str, destination_path: str) -> str:
    """
    Downloads a file from GCS into local path.
    """
    blob = bucket.blob(blob_name)
    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(destination_path)
    return destination_path


def file_exists_in_gcs(blob_name: str) -> bool:
    """
    Check if file exists in GCS.
    """
    return bucket.blob(blob_name).exists()
