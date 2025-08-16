import os
from fastapi import FastAPI

# CRITICAL: Use Railway's PORT environment variable
PORT = int(os.getenv("PORT", 8000))
app = FastAPI()

# CRITICAL: Add health endpoints for Railway
@app.get("/")
async def root():
    return {"status": "healthy", "port": PORT}

@app.get("/health") 
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # CRITICAL: Bind to 0.0.0.0, not localhost
    uvicorn.run(app, host="0.0.0.0", port=PORT)
