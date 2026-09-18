# Python 3.12, slimmest runtime.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero (caché de build eficiente).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && \
    pip install . && \
    pip install ".[dev]"

# Usuario no root por seguridad.
RUN useradd --create-home predictor && \
    chown -R predictor /app
USER predictor

ENTRYPOINT ["predictor"]
CMD ["--help"]