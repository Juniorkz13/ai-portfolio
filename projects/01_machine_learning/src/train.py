import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .data_preprocessing import (
    load_data,
    split_features_target,
    build_preprocessor,
    train_test_data,
)


DATA_PATH = "projects/01_machine_learning/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"


MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    ),
}



def train_models():
    """
    Train machine learning models and save them to disk.
    """

    # 1. Load and prepare data
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_data(X, y)

    # 3. Build preprocessing pipeline
    preprocessor = build_preprocessor(X)

    # 4. Train models
    for name, model in MODELS.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        # 5. Save trained model
        joblib.dump(
            pipeline,
            f"projects/01_machine_learning/models/{name}.joblib",
        )

        print(f"Model '{name}' trained and saved successfully.")


if __name__ == "__main__":
    train_models()
