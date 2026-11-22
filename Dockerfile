# Tax Agent web service. Build and run with docker compose (see docs/deploy.md).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TAX_AGENT_INTERVIEW_DB_PATH=/data/interviews.sqlite3

WORKDIR /app
COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

# Only what the service reads at runtime: code, playbooks, settings, prebuilt guide passages.
COPY src ./src
COPY playbooks ./playbooks
COPY config ./config
COPY data/processed/guide_evidence.json ./data/processed/guide_evidence.json

# Run as an unprivileged user; interviews live on the /data volume.
RUN useradd --create-home --uid 10001 app && mkdir -p /data /app/logs && chown app /data /app/logs
USER app
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

# One worker on purpose: chat sessions, staged uploads and the model-call counters live in memory.
# --proxy-headers: behind Caddy the app sees https (secure cookies) and the visitor's real IP address.
CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
