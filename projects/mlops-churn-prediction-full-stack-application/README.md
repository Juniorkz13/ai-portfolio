
# MLOps Churn Prediction – Full Stack Application

This project is a **full-stack MLOps application** for **customer churn prediction**, combining a machine learning pipeline with a production-ready backend API and a modern frontend interface.

The goal is to demonstrate **end-to-end ML deployment**, from model inference to a user-facing web application.

---

## 🚀 Project Overview

- **Machine Learning**: Trained churn prediction model (scikit-learn pipeline)
- **Backend**: FastAPI REST API for inference
- **Frontend**: React + Vite application for interactive predictions
- **MLOps Focus**: Clear API contract, reproducible setup, and production-ready structure

---

## 🧠 Churn Prediction Model

The model predicts:
- **Churn Probability** (0 → 1)
- **Churn Prediction** (0 = No Churn, 1 = Churn)

### Input Features
| Feature | Description |
|------|------------|
| tenure | Number of months the customer has stayed |
| monthly_charges | Monthly billing amount |
| total_charges | Total amount charged |
| contract_type | Contract duration (month-to-month, one-year, two-year) |
| payment_method | Payment method |
| internet_service | Type of internet service |

---

## 🏗️ Architecture

```
┌──────────────┐      HTTP      ┌──────────────┐
│   Frontend   │ ───────────▶ │   FastAPI    │
│ React + Vite │               │  Backend API │
└──────────────┘               └──────────────┘
                                      │
                                      ▼
                              ML Pipeline (.joblib)
```

---

## 📁 Project Structure

```
projects/mlops-churn-prediction-full-stack-application/
│
├── app/                 # FastAPI backend
│   ├── main.py
│   ├── schemas.py
│
├── models/              # Trained ML pipeline
│   └── churn_pipeline_v1.joblib
│
├── frontend/            # React frontend
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
│
├── README.md
```

---

## 🔌 Backend – FastAPI

### Start the API
```bash
uvicorn app.main:app --reload
```

- API URL: `http://127.0.0.1:8000`
- Swagger Docs: `http://127.0.0.1:8000/docs`

### Prediction Endpoint
**POST** `/v1/predict`

Example payload:
```json
{
  "tenure": 12,
  "monthly_charges": 70.5,
  "total_charges": 850.0,
  "contract_type": "month-to-month",
  "payment_method": "credit_card",
  "internet_service": "fiber_optic"
}
```

Example response:
```json
{
  "churn_probability": 0.4886,
  "churn_prediction": 0
}
```

---

## 🎨 Frontend – React + Vite

The frontend provides an interactive interface to:
- Fill customer data
- Submit predictions
- Display churn probability and classification

### Install dependencies
```bash
cd frontend
npm install
```

### Start the frontend
```bash
npm run dev
```

- Frontend URL: `http://localhost:5173`

### Environment Variable
Create a `.env` file inside `frontend/`:

```env
VITE_API_BASE=http://127.0.0.1:8000
```

---

## 🔒 CORS Configuration

CORS is enabled in the backend to allow frontend communication:
- `http://localhost:5173`
- `http://127.0.0.1:5173`

---

## 🧪 Local End-to-End Test

1. Start backend:
   ```bash
   uvicorn app.main:app --reload
   ```
2. Start frontend:
   ```bash
   cd frontend
   npm run dev
   ```
3. Open browser:
   - `http://localhost:5173`
4. Submit data and receive churn prediction.

---

## 🎯 Key Highlights

- Full-stack ML application
- Clean API contract with validation
- Production-ready FastAPI setup
- Modern React frontend
- Clear separation of concerns
- Ideal for **MLOps / ML Engineer / Full-Stack AI Engineer** portfolios

---

## 📌 Author

**José Geraldo do Espírito Santo Júnior**  
📍 Brazil  
🔗 [LinkedIn](https://www.linkedin.com/in/josejunior13/)

---

## 📜 License
This project is for educational and portfolio purposes.
