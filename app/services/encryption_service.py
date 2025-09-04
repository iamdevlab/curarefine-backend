# app/services/encryption_service.py
import os
from cryptography.fernet import Fernet, InvalidToken

# Load the secret key from environment variables
# The application will fail to start if the key is not set, which is a good security practice.
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    raise ValueError("ENCRYPTION_KEY environment variable not set!")

fernet = Fernet(ENCRYPTION_KEY.encode())


def encrypt(data: str) -> bytes:
    """Encrypts a string and returns it as bytes."""
    if not data:
        return b""
    return fernet.encrypt(data.encode())


def decrypt(encrypted_data: bytes) -> str:
    """Decrypts bytes and returns them as a string."""
    if not encrypted_data:
        return ""
    try:
        return fernet.decrypt(encrypted_data).decode()
    except InvalidToken:
        # This can happen if the data is not encrypted or used a different key
        # For security, you might want to log this but not expose details.
        print("Warning: Failed to decrypt data. It might not be encrypted correctly.")
        # Depending on your logic, you might return the original bytes as a string
        # or handle the error more strictly.
        return str(encrypted_data)
