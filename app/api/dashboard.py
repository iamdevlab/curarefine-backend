from fastapi import APIRouter, Depends, HTTPException
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any

# from app.services.security import get_current_user
from app.services.postgres_client import get_db_cursor

# --- NEW: Import the get_projects_for_user function ---
from app.services.postgres_client import (
    get_connection,
    get_projects_for_user,
    get_dashboard_insights,
)

router = APIRouter(tags=["dashboard"])


# --- Dependency for getting a database cursor ---
def get_db_cursor():
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            yield cursor
    finally:
        if conn:
            conn.close()


@router.get("/dashboard")
async def get_dashboard_data(
    current_user: dict = Depends(get_current_user),
    cursor: RealDictCursor = Depends(get_db_cursor),
):
    """
    Endpoint to fetch all data needed for the main dashboard view.
    """
    user_id = current_user["user_id"]
    try:
        # --- Fetch the list of recent projects ---
        recent_projects = get_projects_for_user(user_id, cursor)
        insights = get_dashboard_insights(user_id, cursor)

        # --- Categorize projects (Example Logic) ---
        # This logic can be expanded to be more sophisticated
        completed_projects = [p for p in recent_projects if p["status"] == "Completed"]
        pending_projects = [p for p in recent_projects if p["status"] != "Completed"]

        return {
            "status": "success",
            "data": {
                "recent_projects": recent_projects,
                "completed_projects": completed_projects,
                "pending_projects": pending_projects,
                "insights": insights,
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching dashboard data: {str(e)}"
        )
