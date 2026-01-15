
# 🚀 Customer Churn Prediction API — MLOps & Deployment

This project demonstrates an end-to-end **Machine Learning deployment pipeline**, covering
model training, preprocessing, inference, and containerized deployment using **FastAPI** and **Docker**.

The objective is to expose a **production-ready churn prediction service** with a clear
feature contract, robust preprocessing, and versioned inference endpoints.

---

## 🧠 Project Overview

Customer churn prediction is a critical business problem for subscription-based companies.
In this project, a trained Machine Learning model is deployed as a REST API, enabling
real-time churn prediction from structured customer data.

This project focuses not only on modeling, but also on **production concerns**, such as:
- Feature schema alignment
- Robust preprocessing
- Inference reliability
- Deployment reproducibility

---

## 🏗️ Architecture

```
Client (JSON Request)
        ↓
FastAPI (/v1/predict)
        ↓
Pydantic Validation
        ↓
Pandas DataFrame
        ↓
Preprocessing Pipeline (Scaling + One-Hot Encoding)
        ↓
Trained ML Model
        ↓
Prediction Response (JSON)
```

---

## 📦 Tech Stack

- Python 3.10
- FastAPI
- Scikit-learn
- Pandas
- Joblib
- Docker
- Docker Compose

---

## 📁 Project Structure

```
projects/mlops_deploy/
├── app/
│   ├── main.py          # FastAPI application
│   ├── schemas.py       # Request/response schemas
│   └── __init__.py
├── models/
│   └── churn_pipeline_v1.joblib
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ▶️ Running the API Locally

From the root of the repository:

```bash
uvicorn projects.mlops_deploy.app.main:app --reload
```

Open:
- Swagger UI: http://127.0.0.1:8000/docs

---

## 🐳 Running with Docker

### Build and run manually
```bash
docker build -f projects/mlops_deploy/Dockerfile -t churn-api projects/mlops_deploy
docker run -p 8000:8000 churn-api
```

### Using Docker Compose
```bash
docker compose up --build
```

---

## 🔮 Prediction Endpoint

**POST** `/v1/predict`

### Example Request
```json
{
  "tenure": 12,
  "monthly_charges": 75.5,
  "total_charges": 900.0,
  "contract_type": "Month-to-month",
  "payment_method": "Electronic check",
  "internet_service": "Fiber optic"
}
```

### Example Response
```json
{
  "churn_probability": 0.8396,
  "churn_prediction": 1
}
```

---

## 📌 MLOps Highlights

- Explicit feature contract between training and inference
- Robust handling of unseen categorical values
- Versioned API endpoint (`/v1/predict`)
- Separation of training and serving logic
- Fully containerized and reproducible deployment

---

## 🚀 Future Improvements

- Model monitoring and drift detection
- Advanced model versioning
- CI/CD pipeline
- Cloud deployment (AWS, GCP, Azure)

---

## 👤 Author

**José Geraldo do Espírito Santo Júnior**  
AI & Machine Learning Portfolio  
Location: Brazil
