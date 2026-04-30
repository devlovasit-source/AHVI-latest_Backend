FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for image processing and vision models
# Using libgl1 instead of libgl1-mesa-glx for modern Debian compatibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the default local port; Railway injects PORT at runtime.
EXPOSE 8080

# Command to run the FastAPI application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
