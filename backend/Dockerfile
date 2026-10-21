# syntax=docker/dockerfile:1.7
# Build context is the repository root (see docker-compose.yaml) so the
# frontend can be baked into the image.

FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY backend/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/requirements.txt


FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FRONTEND_DIR=/app/frontend

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home app

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY backend/main.py /app/main.py
COPY frontend/index.html /app/frontend/index.html

USER app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
