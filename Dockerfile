FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgl1 / libglib2.0-0 are pulled by opencv-python-headless on some CV paths.
# Keep curl + ca-certificates for healthcheck / TLS chain.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Run as non-root for defence-in-depth.
RUN groupadd --system --gid 1000 ahvi \
    && useradd --system --uid 1000 --gid ahvi --create-home ahvi

COPY --chown=ahvi:ahvi . .

USER ahvi

ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8080}/health" || exit 1

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]