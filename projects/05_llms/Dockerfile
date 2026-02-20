FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Working dir
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install python deps
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose API port
EXPOSE 8000

# Start API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]