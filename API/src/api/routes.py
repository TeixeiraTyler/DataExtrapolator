from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from dependency_injector.wiring import inject
from typing import Dict, Any

from src.services.ml_service import MLService
from src.container import Container

import base64

router = APIRouter()


@router.post("/process-csv", response_model=Dict[str, Any])
@inject
async def process_csv(
    file: UploadFile = File(...),
    dependent_variable: str = Form(...),
    num_features: int = Form(...),
    ml_service: MLService = Depends(lambda: Container.ml_service())
):
    """
    Process a CSV file and train machine learning models.
    
    Args:
        file: CSV file to process
        dependent_variable: Name of the target variable/column for prediction
        
    Returns:
        Dictionary with model results and metrics
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read file content
        content = await file.read()
        
        # Process CSV with ML service
        result = ml_service.process_csv(content, dependent_variable, num_features)
        
        # Convert binary plot data to base64 string
        if "correlation_plot" in result and result["correlation_plot"]:
            result["correlation_plot"] = base64.b64encode(result["correlation_plot"]).decode('utf-8')
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}") 
    