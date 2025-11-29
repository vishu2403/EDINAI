FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend:$PYTHONPATH

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential libpq-dev poppler-utils tesseract-ocr && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY backend ./backend

EXPOSE 8000

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
