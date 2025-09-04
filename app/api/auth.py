# app/api/auth.py
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from app.services import security
from app.services.postgres_client import get_user_by_username, get_connection
from psycopg2.extras import RealDictCursor
from app.api.dashboard import get_db_cursor
from pydantic import BaseModel

# In a real app, you'd have a function to get a user from the DB
# from app.services.db import get_user_by_username

router = APIRouter(prefix="/auth", tags=["authentication"])


class UserCreate(BaseModel):
    full_name: str
    email: str
    username: str
    password: str


@router.post("/register")
async def register_user(user_data: UserCreate):
    # Check if the username or email already exists
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT id FROM users WHERE username = %s OR email = %s",
                (user_data.username, user_data.email),
            )
            if cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username or email already registered",
                )

        # Hash the password
        hashed_password = security.get_password_hash(user_data.password)

        # Insert the new user into the database
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (username, full_name, email, hashed_password)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    user_data.username,
                    user_data.full_name,
                    user_data.email,
                    hashed_password,
                ),
            )

    return {"message": "User registered successfully"}


@router.post("/token")
async def login_for_access_token(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    cursor: RealDictCursor = Depends(get_db_cursor),  # Add DB cursor dependency
):
    # --- FIX: Replace the mock user with a real database lookup ---
    user = get_user_by_username(form_data.username, cursor)

    if not user or not security.verify_password(
        form_data.password, user["hashed_password"]
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # --- END FIX ---

    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": str(user["id"])},  # Use the real user ID from the database
        expires_delta=access_token_expires,
    )

    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite="lax",
        secure=False,  # Set to True in production (HTTPS)
    )
    return {"message": "Login successful"}


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(key="access_token")
    return {"message": "Logout successful"}


@router.get("/users/me")
async def read_users_me(current_user: dict = Depends(security.get_current_user)):
    # This endpoint is now protected.
    # The `get_current_user` dependency will run first.
    return current_user
