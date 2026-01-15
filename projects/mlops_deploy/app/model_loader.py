import joblib

MODEL_PATH = "projects/mlops_deploy/models/churn_model.joblib"


def load_model():
    return joblib.load(MODEL_PATH)
