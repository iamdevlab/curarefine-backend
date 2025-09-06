# app/api/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from app.services import security
from app.services.postgres_client import get_user_by_username, get_connection
from psycopg2.extras import RealDictCursor
from app.api.dashboard import get_db_cursor
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["authentication"])


class UserCreate(BaseModel):
    full_name: str
    email: str
    username: str
    password: str


@router.post("/register")
async def register_user(user_data: UserCreate):
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

        hashed_password = security.get_password_hash(user_data.password)

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
    form_data: OAuth2PasswordRequestForm = Depends(),
    cursor: RealDictCursor = Depends(get_db_cursor),
):
    user = get_user_by_username(form_data.username, cursor)

    if not user or not security.verify_password(
        form_data.password, user["hashed_password"]
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": str(user["id"])},
        expires_delta=access_token_expires,
    )

    # Return the token in the response body, as expected by the frontend
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/logout")
async def logout():
    # In a token-based system, logout is handled by the client (deleting the token)
    # This endpoint can remain for completeness if needed
    return {"message": "Logout successful on client"}


@router.get("/users/me")
async def read_users_me(current_user: dict = Depends(security.get_current_user)):
    # This endpoint is now correctly protected by the header-based dependency
    return current_user
