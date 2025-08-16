import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import router as api_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: Get port from Railway environment
PORT = int(os.getenv("PORT", 8000))

# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst Agent",
    description="Multi-modal data analysis agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CRITICAL: Root endpoint for Railway healthcheck
@app.get("/")
async def root():
    return {
        "message": "Data Analyst Agent API",
        "status": "healthy",
        "port": PORT
    }

# Additional health endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "healthy"}

# Startup event for logging
@app.on_event("startup")
async def startup_event():
    logger.info(f"🚀 Data Analyst Agent starting on port {PORT}")
    logger.info("✅ Ready to accept requests!")

# Include your API routes
app.include_router(api_router)

# CRITICAL: Run with correct host and port
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting uvicorn on 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
