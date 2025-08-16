import uvicorn
import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from api.endpoints import router as api_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Render-specific configuration
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/workspaces")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0") 
APP_PORT = int(os.getenv("PORT", 8000))

# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst Agent",
    description="Multi-modal data analysis agent",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ADD THIS: Detailed error handler for 422 debugging
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"❌ Validation Error Details:")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request method: {request.method}")
    logger.error(f"Request headers: {dict(request.headers)}")
    logger.error(f"Validation errors: {exc.errors()}")
    logger.error(f"Request body: {exc.body}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": str(exc.body) if exc.body else None,
            "message": "Validation failed - check parameter names and types"
        }
    )

# Attach API routes
app.include_router(api_router)

# Your existing code...

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
