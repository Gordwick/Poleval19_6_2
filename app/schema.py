from enum import Enum
from pydantic import BaseModel, Field
from app_config import CONFIG
    
Choices = Enum("Choices", { c:c for c in CONFIG["CATEGORIZATION_CLASSES"]})

class PredictionInput(BaseModel):
    """ Text input for model """
    message: str


class PredictionResult(BaseModel):
    """Prediction result from the model"""
    prediction: Choices = Field(..., example='cyberbullying', title='Final model prediction')


class PredictionResponse(BaseModel):
    """Output response for prediction"""
    error: bool = Field(..., example=False, title='Whether there is error')
    results: PredictionResult = ...


class ErrorResponse(BaseModel):
    """Error response for the API"""
    error: bool = Field(..., example=True, title='Whether there is error')
    message: str = Field(..., example='', title='Error message')