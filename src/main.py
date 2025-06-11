# /home/ubuntu/open_notebook_full_backend/fastapi_backend/src/main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager

from .database import connect_db, close_db, SURREAL_URL, SURREAL_NS, SURREAL_DB
from .routers import notebooks, notes, sources, ai_interactions, podcasts, search, models

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("Application startup...")
    await connect_db() # Establish DB connection
    yield
    # Code to run on shutdown
    print("Application shutdown...")
    await close_db() # Close DB connection

# Create the FastAPI app instance with lifespan management
app = FastAPI(
    title="Open Notebook Backend API",
    description="API providing backend functionality for the Open Notebook application, mirroring Streamlit UI features.",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers from different modules
app.include_router(notebooks.router)
app.include_router(notes.router)
app.include_router(sources.router)
app.include_router(ai_interactions.router)
app.include_router(podcasts.router)
app.include_router(search.router)
app.include_router(models.router)

# Simple root endpoint for health check / info
@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the Open Notebook Backend API!",
        "database_status": f"Connected to {SURREAL_URL} (NS: {SURREAL_NS}, DB: {SURREAL_DB})" if notebooks.db else "Database connection failed",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json"
    }

# Placeholder for task status endpoint (if needed globally)
# @app.get("/api/v1/tasks/{task_id}", tags=["Tasks"])
# async def get_task_status(task_id: str):
#     # Logic to check task status from a background task queue (e.g., Celery, ARQ)
#     return {"task_id": task_id, "status": "unknown"}

print("FastAPI application instance created.")

