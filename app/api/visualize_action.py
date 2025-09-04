# app/api/visualize_action.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

# --- FIX 1: Import the real service from visualizer.py ---
# This was incorrectly pointing to visualize_api.py
from app.visualize.visualizer import VisualizationService
from app.services.security import get_current_user


# Pydantic model for request body validation
class VisualizationRequest(BaseModel):
    rows: List[Dict[str, Any]]
    domain: str = "generic"


# Router setup
router = APIRouter()
logger = logging.getLogger("visualization")


@router.post("/visualize", tags=["visualization"])
async def create_visualizations(
    request: VisualizationRequest, current_user: dict = Depends(get_current_user)
):
    """
    Receives table data, generates chart specifications using the VisualizationService,
    and returns them as a list of JSON strings.
    """
    logger.info(f"Received visualization request for domain: {request.domain}")

    if not request.rows:
        raise HTTPException(
            status_code=400, detail="Cannot generate charts: 'rows' data is empty."
        )

    try:
        # This now calls the correct service which returns JSON specs
        chart_specs = VisualizationService.generate_visualizations(
            table_data=request.rows, domain=request.domain
        )

        # --- FIX 2: Return the "charts" key for the interactive frontend ---
        # This was incorrectly returning "chart_urls"
        return {
            "message": "Successfully generated chart specifications.",
            "charts": chart_specs,
        }

    except Exception as e:
        logger.error(f"Error during chart generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
