FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only essential files for now
COPY main.py .
COPY api/ ./api/
COPY agent/ ./agent/

# Create workspace directory
RUN mkdir -p temp_workspaces && chmod 755 temp_workspaces

# Don't create custom user for now - run as root to eliminate variables
# RUN useradd -m appuser && chown -R appuser:appuser /app
# USER appuser

# Critical: Log the port and start uvicorn properly
CMD ["sh", "-c", "echo 'Container starting on PORT:' $PORT && exec uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info --access-log"]
