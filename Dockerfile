# Dockerfile

# Use official Playwright image with Python and browsers installed
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Playwright dependencies are pre-installed in this image, no additional commands required
