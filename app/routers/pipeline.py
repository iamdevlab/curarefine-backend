# In app/routers/pipeline.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from app.api.clean import get_or_create_cleaner
from app.deep_analysis.cleaning_pipeline import DatacuraPipeline


# Define the request model for the frontend
class PipelineRequest(BaseModel):
    file_id: str
    user_id: int
    parameters: Dict[str, Any] = {}


# Setup the router
router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


@router.post("/run")
async def run_automated_pipeline(request: PipelineRequest):
    """
    Instantiates and runs the full DatacuraPipeline with user-defined parameters.
    """
    try:
        # 1. Get the most current state of the data
        cleaner = get_or_create_cleaner(request.user_id, request.file_id)
        df = cleaner.df

        # 2. Initialize the pipeline with the current data
        pipeline = DatacuraPipeline(df=df)

        # 3. Run the full pipeline with parameters from the frontend request
        # The frontend will send parameters like:
        # { "outlier_action": "cap", "missing_strategy": "fill_median" }
        final_report = pipeline.run_pipeline(**request.parameters)

        # 4. The pipeline modifies the dataframe in the cleaner object.
        # We need to update the session with the newly cleaned data.
        # (Assuming a function `update_session` exists or modifying the cleaner)
        cleaner.df = pipeline.cleaner.df  # Ensure the cleaner's df is the final one
        # Here you would ideally save the cleaner's new state to your session cache (e.g., Redis)

        return {
            "status": "success",
            "message": "Pipeline completed successfully.",
            "report": final_report,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running pipeline: {str(e)}")
