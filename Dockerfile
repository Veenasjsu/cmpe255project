FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies for psycopg2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc \
      libpq-dev \
      python3-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy your code and assets
COPY src/ ./src
COPY assets/ ./assets

EXPOSE 8050
CMD ["gunicorn", "Analysis:app", "--bind", "0.0.0.0:8050"]
