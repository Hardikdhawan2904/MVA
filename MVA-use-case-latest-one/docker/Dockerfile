FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy all application code first (needed for pip install)
COPY pyproject.toml .
COPY app/ ./app/
COPY config/ ./config/
COPY migrations/ ./migrations/
COPY alembic.ini .

# Install Python dependencies (non-editable)
RUN pip install --no-cache-dir ".[dev]"

# Create temp directory
RUN mkdir -p /app/tmp/uploads

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
