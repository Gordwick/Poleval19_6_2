import uvicorn
from fastapi import FastAPI, Request
from fastapi.logger import logger
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

import torch
import numpy as np
from transformers import BertConfig, AutoTokenizer

from model import MyHerBertaModel
from predict import Predictor
from schema import PredictionResult
from app_config import CONFIG
from exception_handler import validation_exception_handler, python_exception_handler
from schema import *

# Initialize API Server
app = FastAPI(
    title="HerBerta Classification model",
    description="Model categorizes texts into 3 categories: non-harmful, cyberbullying, hate-speech",
    version="0.0.1",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Load custom exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)


@app.on_event("startup")
async def startup_event():
    """Initialize FastAPI and HerBerta model with Tokenizer and Predictor"""

    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    # Initialize the pytorch model
    model_config = BertConfig.from_pretrained(CONFIG['PRETRAINED_MODEL'])
    model = MyHerBertaModel(conf=model_config)
    
    model.load_state_dict(torch.load(
        CONFIG['MODEL_PATH'], map_location=torch.device(CONFIG['DEVICE'])))
    model.eval()

    #tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['PRETRAINED_MODEL'])
    
    #Predictor
    predictor = Predictor(tokenizer, model)
    
    # add model and other preprocess tools too app state
    app.package = {
        "tokenizer": tokenizer,
        "model": model,
        "predictor": predictor
    }


@app.post('/api/v1/predict',
          response_model=PredictionResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
def do_predict(request: Request, body: PredictionInput):
    """Prediction endpoint"""

    logger.info('API predict called.')
    logger.info(f'input: {body}')

    Predictor = app.package['predictor']
    
    # prepare input data
    X: str = body.message
    y: np.ndarray = Predictor.predict(X)

    # generate prediction based on probablity
    pred = CONFIG['CATEGORIZATION_CLASSES'][y.argmax()]
    logger.info(f"Prediction: {pred}")

    results = PredictionResult(
        prediction=pred
        )
    
    logger.info(f'Results: {results}')

    return PredictionResponse(error=False, results=results)


if __name__ == '__main__':
    uvicorn.run("run_server:app", host="0.0.0.0", port=8080,
                reload=True, log_config="log.ini"
                )