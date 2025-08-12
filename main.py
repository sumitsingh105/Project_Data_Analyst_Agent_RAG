import uvicorn
import os
from fastapi import FastAPI
from api.endpoints import router as api_router

# Render-specific configuration
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/workspaces")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0") 
APP_PORT = int(os.getenv("PORT", 10000))  # Render uses PORT

# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst Agent",
    description="Multi-modal data analysis agent",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Attach API routes
app.include_router(api_router)

def startup_tasks():
    print("Initializing Data Analyst Agent...")
    
    # Skip Docker/sandbox setup on Render
    print("Running on Render - skipping local setup")
    
    # Create temp directory
    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"Temporary workspace directory: '{TEMP_DIR}'")

@app.on_event("startup")
async def startup_event():
    startup_tasks()

@app.get("/")
async def root():
    return {
        "message": "Data Analyst Agent API", 
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "Data Analyst Agent",
        "capabilities": ["Wikipedia Analysis", "DuckDB Court Analysis"]
    }

if __name__ == "__main__":
    startup_tasks()
    print(f"Starting FastAPI server at http://{APP_HOST}:{APP_PORT}")
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
