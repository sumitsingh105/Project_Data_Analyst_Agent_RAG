import os
import logging
from fastapi import FastAPI

# Simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Get Railway port
PORT = int(os.getenv("PORT", 8000))

# Minimal FastAPI app
app = FastAPI(title="Data Analyst Agent")

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"status": "healthy", "message": "Data Analyst Agent is running", "port": PORT}

@app.get("/health")  
async def health():
    logger.info("Health endpoint called")
    return {"status": "ok"}

@app.on_event("startup")
async def startup():
    logger.info(f"🚀 FastAPI starting on 0.0.0.0:{PORT}")
    
@app.on_event("shutdown") 
async def shutdown():
    logger.info("👋 FastAPI shutting down")

# Only import your routes if the basic app works
# from api.endpoints import router as api_router  
# app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting uvicorn server on 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
