FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

WORKDIR /app
COPY requirements.txt .

# Install system dependencies (tesseract for OCR)
RUN apt-get update && apt-get install -y \
    tesseract-ocr tesseract-ocr-eng libtesseract-dev \
    libgl1-mesa-glx libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install chromium && playwright install-deps chromium

COPY . .
RUN mkdir -p temp_workspaces && chmod 755 temp_workspaces

# CRITICAL: Use Railway's PORT environment variable
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info"]
