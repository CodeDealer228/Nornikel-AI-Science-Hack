# Nornikel Knowledge Graph — base image
# Used for both the API server and the ingestion script.
FROM python:3.12-slim

# System dependencies. libffi + libxml2 are needed for some Python
# packages; build-essential is here for pip wheels that have no
# prebuilt for Python 3.12. curl is for HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libffi-dev \
        libxml2-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory.
WORKDIR /app

# Install Python dependencies first (better layer caching).
COPY requirements.txt* pyproject.toml* setup.py* ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; \
    elif [ -f pyproject.toml ]; then pip install --no-cache-dir .; \
    else echo "No requirements file found; install manually."; fi

# Copy the rest of the application.
COPY . .

# Default environment.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO \
    LOG_FORMAT=text

# Default command (overridden in docker-compose per service).
CMD ["python", "-m", "agent.cli", "--help"]
