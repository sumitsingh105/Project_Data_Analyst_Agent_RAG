import uvicorn
import os
from fastapi import FastAPI
from api.endpoints import router as api_router
from agent.sandbox import create_sandbox_image

# Environment-configurable workspace directory
TEMP_DIR = os.getenv("TEMP_DIR", "temp_workspaces")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))

# Initialize FastAPI app
app = FastAPI(title="Data Analyst Agent")

# Attach API routes
app.include_router(api_router)

def startup_tasks():
    print("Initializing Data Analyst Agent...")
    create_sandbox_image()
    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"Temporary workspace directory is '{TEMP_DIR}'.")

if __name__ == "__main__":
    startup_tasks()
    print(f"Starting FastAPI server at http://{APP_HOST}:{APP_PORT}")
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
