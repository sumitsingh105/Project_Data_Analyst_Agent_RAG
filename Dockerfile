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

# CRITICAL: Use Railway's PORT environment variable
EXPOSE $PORT

# CRITICAL: Start command that uses Railway PORT
CMD ["sh", "-c", "echo 'Starting on port:' $PORT && uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info"]
