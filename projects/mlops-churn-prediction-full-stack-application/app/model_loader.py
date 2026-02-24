import joblib

MODEL_PATHS = {
    "v1": "projects/mlops_deploy/models/churn_pipeline_v1.joblib",
    # "v2": "projects/mlops_deploy/models/churn_pipeline_v2.joblib",
}

_loaded_models = {}


def load_model(version: str):
    if version not in MODEL_PATHS:
        raise ValueError(f"Model version '{version}' not available")

    if version not in _loaded_models:
        _loaded_models[version] = joblib.load(MODEL_PATHS[version])

    return _loaded_models[version]
