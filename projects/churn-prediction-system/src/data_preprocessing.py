import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data(path: str) -> pd.DataFrame:
    """
    Load and clean the churn dataset.
    """
    df = pd.read_csv(path)

    # Fix TotalCharges type (comes as string)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with missing values
    df = df.dropna()

    return df


def split_features_target(df):
    df = df.rename(columns={
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "Contract": "contract_type",
        "PaymentMethod": "payment_method",
        "InternetService": "internet_service",
    })

    FEATURES = [
        "tenure",
        "monthly_charges",
        "total_charges",
        "contract_type",
        "payment_method",
        "internet_service",
    ]

    X = df[FEATURES]
    y = df["Churn"].map({"Yes": 1, "No": 0})

    return X, y




def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline for numerical and categorical features.
    """
    categorical_cols = X.select_dtypes(include="object").columns
    numerical_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return preprocessor


def train_test_data(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split data into train and test sets with stratification.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
