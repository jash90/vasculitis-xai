# Dockerfile for Vasculitis XAI API
FROM python:3.11-slim

# Metadane
LABEL maintainer="Vasculitis XAI Project"
LABEL description="API do predykcji śmiertelności w zapaleniu naczyń"
LABEL version="1.0.0"

# Ustaw zmienne środowiskowe
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Ustaw katalog roboczy
WORKDIR /app

# Zainstaluj zależności systemowe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Kopiuj requirements i zainstaluj zależności Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Kopiuj kod aplikacji
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Utwórz użytkownika nie-root
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check (Railway wstrzykuje PORT; lokalnie domyślnie 8000)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Uruchom API — bind do PORT z platformy (Railway), fallback 8000
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
