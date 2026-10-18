FROM python:3.11-slim

# Tesseract is the OCR engine that pytesseract calls.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000
CMD gunicorn --bind 0.0.0.0:${PORT} --timeout 120 app:app
