FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLOODSIGHT_CACHE_DIR=/data/cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY floodsight ./floodsight

RUN useradd --create-home --uid 1000 floodsight \
    && mkdir -p /data/cache \
    && chown -R floodsight /data
USER floodsight

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "floodsight", "--host", "0.0.0.0", "--port", "8000"]
