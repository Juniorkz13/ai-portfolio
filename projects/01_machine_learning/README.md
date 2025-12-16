# Customer Churn Prediction — Telecom Dataset

## 📌 Project Overview

Customer churn is one of the most critical problems for subscription-based businesses, such as telecom companies.
This project aims to predict customer churn using classical Machine Learning techniques, enabling proactive retention strategies and reducing revenue loss.

The solution follows a complete end-to-end ML pipeline, from exploratory data analysis to model tuning and evaluation.

---

## 💼 Business Problem

Acquiring new customers is significantly more expensive than retaining existing ones.
Predicting which customers are likely to churn allows companies to:

-   Prioritize retention campaigns
-   Optimize marketing costs
-   Improve customer lifetime value (CLV)

---

## 📂 Dataset

-   Source: Kaggle — Telco Customer Churn
-   File: WA*Fn-UseC*-Telco-Customer-Churn.csv
-   Target variable: Churn (Yes / No)

The dataset contains customer demographics, service usage, contract details, and billing information.

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights obtained during EDA:

-   The dataset is imbalanced, with churn being the minority class.
-   Customers with short tenure show significantly higher churn rates.
-   Month-to-month contracts are strongly associated with churn.
-   Payment method and internet service type have a clear influence on customer cancellation.

EDA was essential to guide preprocessing decisions and model strategy.

---

## 🛠️ Data Preprocessing

The preprocessing pipeline includes:

-   Handling missing values
-   Conversion of numerical features (TotalCharges)
-   Feature scaling for numerical variables
-   One-hot encoding for categorical variables

All preprocessing steps are integrated into a Scikit-learn Pipeline, ensuring reproducibility and preventing data leakage.

---

## 🤖 Models Trained

The following models were implemented and compared:

-   Logistic Regression
-   Random Forest Classifier

To address class imbalance, class weighting was applied to both models.

---

## 🔧 Hyperparameter Tuning & Cross-Validation

Model optimization was performed using GridSearchCV with:

-   5-fold stratified cross-validation
-   ROC-AUC as the primary evaluation metric

This ensures robust generalization and avoids overfitting.

---

## 📈 Model Evaluation

Models were evaluated using:

-   Precision, Recall, and F1-score
-   ROC-AUC
-   Confusion Matrix
-   ROC Curve comparison

Since churn prediction is a business-critical task, Recall was prioritized to minimize false negatives (customers who churn but are not identified).

Additionally, the decision threshold was adjusted to better align with business objectives.

---

## 🧪 Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Seaborn
-   Joblib

---

## ▶️ How to Run the Project

From the root of the repository, run:

```bash
python -m projects.01_machine_learning.src.train
python -m projects.01_machine_learning.src.evaluate
```

## 📊 Results and Conclusion

The tuned models demonstrated strong predictive performance, especially after:

-   Handling class imbalance
-   Applying hyperparameter tuning
-   Optimizing decision thresholds

This project shows that classical machine learning models, when properly designed and evaluated, can effectively solve real-world business problems such as customer churn prediction.

---

## 🚀 Next Steps

Possible future improvements include:

-   Gradient boosting models (XGBoost / LightGBM)
-   Cost-sensitive evaluation
-   Model deployment as an API or dashboard

---

## 👤 Author

José Geraldo do Espírito Santo Júnior
AI and Machine Learning Portfolio
Location: Brazil

```

```
