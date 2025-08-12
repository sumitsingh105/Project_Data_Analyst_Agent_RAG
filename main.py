import uvicorn
import os
from fastapi import FastAPI
from api.endpoints import router as api_router
from agent.sandbox import create_sandbox_image

# Render uses PORT 10000 by default
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/workspaces")  # Use /tmp for Render
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 10000))  # Render default port

app = FastAPI(title="Data Analyst Agent")
app.include_router(api_router)

def startup_tasks():
    print("Initializing Data Analyst Agent...")
    try:
        create_sandbox_image()
    except Exception as e:
        print(f"Sandbox creation failed (expected on Render): {e}")
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"Temporary workspace directory: '{TEMP_DIR}'")

if __name__ == "__main__":
    startup_tasks()
    print(f"Starting FastAPI server at http://{APP_HOST}:{APP_PORT}")
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
