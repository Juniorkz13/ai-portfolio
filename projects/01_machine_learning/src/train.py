import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from .data_preprocessing import (
    load_data,
    split_features_target,
    build_preprocessor,
    train_test_data,
)

DATA_PATH = "projects/01_machine_learning/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

PARAM_GRIDS = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"],
    },
    "random_forest": {
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    },
}

MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    ),
    "random_forest": RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    ),
}


def train_models():
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_data(X, y)
    preprocessor = build_preprocessor(X)

    for name, model in MODELS.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        grid_search = GridSearchCV(
            pipeline,
            PARAM_GRIDS[name],
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        print(f"\nBest parameters for {name}:")
        print(grid_search.best_params_)
        print("Best CV ROC-AUC:", grid_search.best_score_)

        if name == "logistic_regression":
            joblib.dump(
                grid_search.best_estimator_,
                "projects/01_machine_learning/models/churn_pipeline.joblib",
            )
            print("Production churn pipeline saved.")


if __name__ == "__main__":
    train_models()
