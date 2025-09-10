# app/api/visualize_action.py

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.services.security import get_current_user
from app.visualize.visualizer import VisualizationService


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
    and returns them as a list of structured chart objects.
    """
    logger.info(f"Received visualization request for domain: {request.domain}")

    if not request.rows:
        raise HTTPException(
            status_code=400, detail="Cannot generate charts: 'rows' data is empty."
        )

    try:
        # This calls the service which returns a list of chart specification objects
        chart_specs = VisualizationService.generate_visualizations(
            table_data=request.rows, domain=request.domain
        )

        # 💡 FIX: Return the full chart specification objects directly.
        # Do not extract just the 'spec' property.
        return {
            "message": "Successfully generated chart specifications.",
            "charts": chart_specs,
        }

    except Exception as e:
        logger.error(f"Error during chart generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")