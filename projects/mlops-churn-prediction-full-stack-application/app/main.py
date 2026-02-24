from fastapi import FastAPI
import pandas as pd
import joblib

from app.schemas import ChurnInput, PredictionOutput
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/churn_pipeline_v1.joblib"
pipeline = joblib.load(MODEL_PATH)

EXPECTED_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "payment_method",
    "internet_service",
]

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/v1/predict", response_model=PredictionOutput)
def predict_v1(input: ChurnInput):
    df = pd.DataFrame([input.dict()])
    df = df[EXPECTED_FEATURES]

    probability = pipeline.predict_proba(df)[0][1]
    prediction = int(probability >= 0.5)

    return PredictionOutput(
        churn_probability=round(float(probability), 4),
        churn_prediction=prediction,
    )
