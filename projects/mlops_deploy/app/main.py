from fastapi import FastAPI
from projects.mlops_deploy.app.schemas import ChurnInput, PredictionOutput
from projects.mlops_deploy.app.model_loader import load_model

import numpy as np

app = FastAPI(title="Customer Churn Prediction API")

model = load_model()

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running",
        "docs": "http://127.0.0.1:8000/docs"
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_churn(data: ChurnInput):
    features = np.array([[
        data.tenure,
        data.monthly_charges,
        data.total_charges,
    ]])

    probability = model.predict_proba(features)[0][1]
    prediction = int(probability >= 0.5)

    return PredictionOutput(
        churn_probability=probability,
        churn_prediction=prediction,
    )
