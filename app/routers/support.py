# In a new router file, e.g., routers/support.py or directly in your main.py

import os
import resend
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr

# If using a separate router file
# router = APIRouter()

# If in main.py, use 'app' instead of 'router'
router = APIRouter(
    prefix="/api/support",
    tags=["support"]
)

# Load API key from environment
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
if not RESEND_API_KEY:
    raise ValueError("RESEND_API_KEY is not set in environment variables")

resend.api_key = RESEND_API_KEY


class ContactFormRequest(BaseModel):
    fullName: str
    email: EmailStr
    message: str


@router.post("/contact")
async def handle_contact_form(request: ContactFormRequest):
    """
    Receives contact form data and sends it as an email.
    """
    try:
        params = {
            "from": "Support Form <onboarding@resend.dev>",  # Use a verified domain in production
            "to": ["devlabstudios@outlook.com"],
            "subject": f"New Contact Form Submission from {request.fullName}",
            "html": f"""
                <h1>New Support Request</h1>
                <p><strong>Name:</strong> {request.fullName}</p>
                <p><strong>Email:</strong> {request.email}</p>
                <hr>
                <p><strong>Message:</strong></p>
                <p>{request.message}</p>
            """,
            "reply_to": request.email
        }

        email = resend.emails.send(params)

        # You can check email['id'] to see if it was successfully initiated
        if email.get("id"):
            return {"message": "Email sent successfully."}
        else:
            # Handle cases where the email service accepts the request but doesn't return an ID
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=email.get("error", "Failed to send email.")
            )

    except Exception as e:
        print(f"Error sending email: {e}")  # Log the error for debugging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while sending the message."
        )