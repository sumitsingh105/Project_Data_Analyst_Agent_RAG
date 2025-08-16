# Use official Playwright image with Python and browsers installed
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

RUN playwright install chromium
RUN playwright install-deps chromium

COPY . .

RUN mkdir -p temp_workspaces
RUN chmod 755 temp_workspaces

# Skip user creation for now (causing UID conflicts)
# RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
# USER appuser

EXPOSE $PORT

# FIXED: Use Railway's PORT environment variable
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
