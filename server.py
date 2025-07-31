# server.py

import os
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError

# --- Database Configuration ---
# Get the database URL from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set. Please configure it on Render.")

# SQLAlchemy setup
# We'll use a simple declarative base for now, though you might use models later
Base = declarative_base()

# Create a database engine
# The 'pool_pre_ping=True' helps with connection stability on Render
# The connect_args might be needed for SSL if you're connecting from outside Render's private network
# but Render's internal connections usually handle SSL transparently when using the internal URL.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Create a session local class for database interactions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- FastAPI Application ---
app = FastAPI(
    title="The Aboutness Project Backend API",
    description="API for semantic search and RAG for academic papers.",
    version="0.1.0",
)

@app.on_event("startup")
async def startup_event():
    print("Backend application starting up...")
    # Optional: Verify database connection on startup
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("Successfully connected to the database!")
    except OperationalError as e:
        print(f"Database connection failed on startup: {e}")
        # In production, you might want to raise an exception or log this error more critically
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")


@app.get("/")
async def read_root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {"message": "Welcome to The Aboutness Project Backend API!"}

@app.get("/health")
async def health_check(db: SessionLocal = Depends(get_db)):
    """
    Health check endpoint to verify API and database connectivity.
    """
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "database_connection": "successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")

# --- Placeholder for future RAG/Search Endpoints ---
# @app.post("/search-papers")
# async def search_papers(query: str, db: SessionLocal = Depends(get_db)):
#     # This is where your vector database search logic will go
#     # 1. Embed the query
#     # 2. Query pgvector for similar embeddings
#     # 3. Retrieve associated paper metadata
#     # 4. (Optional) Rerank with citation graph
#     # 5. Pass context to LLM for summary
#     return {"results": "Search functionality coming soon!"}

# @app.post("/ingest-paper")
# async def ingest_paper(paper_data: dict, db: SessionLocal = Depends(get_db)):
#     # This is where your paper ingestion logic will go
#     # 1. Parse paper_data (abstract, title, etc.)
#     # 2. Chunk text
#     # 3. Generate embeddings
#     # 4. Store in pgvector and metadata in regular PostgreSQL tables
#     return {"status": "Ingestion functionality coming soon!"}
