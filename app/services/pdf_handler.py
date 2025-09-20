# app/services/pdf_handler.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.report import ReportRequest
# Import the generation function from its correct location in utils
from app.utils.pdfreports import generate_comprehensive_report

# Define the router that will be imported into main.py
router = APIRouter()


@router.post("/generate", response_class=StreamingResponse)
async def generate_report_endpoint(request: ReportRequest):
    """
    API endpoint to generate and return the PDF report.
    This function handles the web request and calls the business logic.
    """
    try:
        if not request.current_data:
            raise HTTPException(status_code=400, detail="No data provided to generate report.")

        # Call the generation function from the utils module
        pdf_bytes_io = generate_comprehensive_report(
            current_data=request.current_data,
            project_name=request.project_name
        )

        filename = f"{request.project_name.replace(' ', '_')}_Report.pdf"

        # Stream the response back to the client
        return StreamingResponse(
            iter([pdf_bytes_io.getvalue()]),
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )

    except Exception as e:
        print(f"Error generating PDF report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while generating the report: {e}"
        )