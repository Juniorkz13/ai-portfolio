from pydantic import BaseModel

class ChurnInput(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    payment_method: str
    internet_service: str


class PredictionOutput(BaseModel):
    churn_probability: float
    churn_prediction: int
