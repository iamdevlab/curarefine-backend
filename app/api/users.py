# In a new file, e.g., app/api/users.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.services.security import get_current_user
from app.services.postgres_client import get_connection
from psycopg2.extras import RealDictCursor
from app.services.security import get_current_user, verify_password, get_password_hash

router = APIRouter(prefix="/users", tags=["users"])


class UserProfile(BaseModel):
    full_name: str
    email: str


class UserProfileUpdate(BaseModel):
    full_name: str
    email: str


class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str


@router.post("/change-password")
async def change_user_password(
    password_data: PasswordUpdate, current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]

    # 1. Fetch the user's current hashed password from the database
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT hashed_password FROM users WHERE id = %s", (user_id,)
            )
            user_record = cursor.fetchone()

    if not user_record:
        raise HTTPException(status_code=404, detail="User not found")

    current_hashed_password = user_record[0]

    # 2. Verify that the provided 'current_password' is correct
    if not verify_password(password_data.current_password, current_hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect current password")

    # 3. Hash the new password and update it in the database
    new_hashed_password = get_password_hash(password_data.new_password)

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET hashed_password = %s WHERE id = %s",
                (new_hashed_password, user_id),
            )

    return {"status": "success", "message": "Password updated successfully"}


@router.get("/profile", response_model=UserProfile)
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    query = "SELECT full_name, email FROM users WHERE id = %s"
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (user_id,))
            user = cursor.fetchone()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            return user


@router.put("/profile")
async def update_user_profile(
    profile_data: UserProfileUpdate, current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    query = "UPDATE users SET full_name = %s, email = %s WHERE id = %s"
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (profile_data.full_name, profile_data.email, user_id))
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="User not found")
    return {"status": "success", "message": "Profile updated successfully"}
