FROM python:3.10.13-slim

WORKDIR /app

COPY projects/legal-rag-multi-agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY projects/legal-rag-multi-agent /app

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]