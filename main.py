import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import router as api_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get port from environment (Railway sets this)
PORT = int(os.getenv("PORT", 8000))

# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst Agent",
    description="Multi-modal data analysis agent with OCR and web scraping",
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

# CRITICAL: Add health check endpoint for Railway
@app.get("/")
async def root():
    return {
        "message": "Data Analyst Agent API", 
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Simple health check for Railway"""
    return {"status": "ok", "service": "healthy"}

# Include your API routes
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
