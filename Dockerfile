# Use official lightweight Python image
FROM python:3.9-slim

# Set system environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off

# Install system dependencies (build-essential for compiling libraries if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy dependency configuration first to leverage Docker layer caching
COPY requirements.txt /app/

# Install python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . /app/

# Ensure workspace folder structures exist
RUN mkdir -p /app/data /app/results

# Volume for data input and results output
VOLUME ["/app/data", "/app/results"]

# Default entry point command
ENTRYPOINT ["python", "main.py"]
