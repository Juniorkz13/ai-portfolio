import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    recall_score
)

from .data_preprocessing import (
    load_data,
    split_features_target,
    train_test_data,
)


DATA_PATH = "projects/01_machine_learning/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

MODELS = ["logistic_regression", "random_forest"]


def evaluate():
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_data(X, y)

    plt.figure(figsize=(8, 6))

    for model_name in MODELS:
        model = joblib.load(
    f"projects/01_machine_learning/models/{model_name}.joblib"
)


        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        threshold = 0.4
        adjusted_preds = (probs >= threshold).astype(int)

        print(f"\nAdjusted threshold: {threshold}")
        print(classification_report(y_test, adjusted_preds))
        print(f"\nModel: {model_name}")
        print(classification_report(y_test, preds))
        recall = recall_score(y_test, preds)
        print("Recall (Churn):", recall)
        print("ROC-AUC:", roc_auc_score(y_test, probs))

        # Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(y_test, preds)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{model_name}")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    evaluate()
