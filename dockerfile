FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for LightGBM, XGBoost, CatBoost
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Upgrade pip and increase timeout for heavy installs
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u", "main.py"]
