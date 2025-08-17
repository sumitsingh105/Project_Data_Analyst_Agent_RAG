import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from api.endpoints import router as api_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/workspaces")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", 10000))

def startup_tasks():
    print("Initializing Data Analyst Agent...")
    print("Running on server - skipping local setup")
    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"Temporary workspace directory: '{TEMP_DIR}'")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    startup_tasks()
    yield
    # Shutdown (if needed)
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Data Analyst Agent",
    description="Multi-modal data analysis agent",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Error handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"❌ Validation Error Details:")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request method: {request.method}")
    logger.error(f"Validation errors: {exc.errors()}")
    
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
        "capabilities": ["Wikipedia Analysis", "CSV Analysis", "DuckDB Court Analysis"]
    }

if __name__ == "__main__":
    startup_tasks()
    print(f"Starting FastAPI server at http://{APP_HOST}:{APP_PORT}")
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
