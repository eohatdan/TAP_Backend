# server.py

import os
import requests
import uuid
import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import create_engine, text, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import OperationalError

# A key dependency for vector embeddings
from pgvector.sqlalchemy import Vector

# --- RAG Specific Dependencies ---
from sentence_transformers import SentenceTransformer

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Embedding Model ---
# This will be loaded once on application startup for efficiency.
embedding_model = None

# --- Database Configuration ---
# Get the database URL from environment variables set on Render
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set. Please configure it on Render.")
    pass

# SQLAlchemy setup
Base = declarative_base()

# Create a database engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Create a session local class for database interactions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Database Models ---
class Paper(Base):
    __tablename__ = 'papers'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String)
    abstract = Column(String)
    authors = Column(String)
    url = Column(String)
    
    embedding = Column(Vector(384))

# --- Dependency to get a database session ---
def get_db():
    """
    Dependency function to provide a database session to API endpoints.
    """
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
    """
    This event handler runs once when the application starts up.
    """
    logger.info("Backend application starting up...")
    
    # 1. Verify database connection
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Successfully connected to the database!")
    except OperationalError as e:
        logger.error(f"Database connection failed on startup: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")

    # 2. Load the embedding model
    global embedding_model
    try:
        logger.info("Loading Sentence-Transformer model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load embedding model.")

    # 3. Create the 'papers' table if it doesn't exist
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema validated/created.")


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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database connection failed: {e}")

@app.post("/ingest-data", status_code=status.HTTP_201_CREATED)
async def ingest_data(query: str, limit: int = 100, db: SessionLocal = Depends(get_db)):
    """
    Fetches papers from Semantic Scholar based on a query, generates embeddings,
    and stores them in the database.
    """
    global embedding_model
    if embedding_model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Embedding model not loaded.")

    logger.info(f"Fetching papers for query: '{query}' with limit={limit}...")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={limit}&fields=title,abstract,authors,url"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch data from Semantic Scholar: {e}")

    papers_to_ingest = []
    if 'data' not in data:
        return {"message": "No papers found for this query."}

    for paper_data in data['data']:
        if paper_data.get('abstract') and paper_data.get('title'):
            authors_list = [author['name'] for author in paper_data.get('authors', [])]
            
            text_to_embed = f"Title: {paper_data['title']}. Abstract: {paper_data['abstract']}"
            
            embedding = embedding_model.encode(text_to_embed).tolist()
            
            paper_entry = Paper(
                title=paper_data['title'],
                abstract=paper_data['abstract'],
                authors=", ".join(authors_list),
                url=paper_data.get('url'),
                embedding=embedding
            )
            papers_to_ingest.append(paper_entry)

    logger.info(f"Ingesting {len(papers_to_ingest)} papers into the database...")
    try:
        db.bulk_save_objects(papers_to_ingest)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database ingestion failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database ingestion failed: {e}")
    
    return {"message": f"Successfully ingested {len(papers_to_ingest)} papers.", "papers_ingested": len(papers_to_ingest)}


@app.post("/bulk-load-gist", status_code=status.HTTP_201_CREATED)
async def bulk_load_gist(gist_id: str, db: SessionLocal = Depends(get_db)):
    """
    Drops all existing data, fetches content from a multi-file Gist,
    and ingests each file as a separate document.
    """
    global embedding_model
    if embedding_model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Embedding model not loaded.")

    # 1. Drop all existing data
    logger.info("Dropping existing 'papers' table to prepare for bulk load...")
    try:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine) # Re-create the table
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to drop/re-create table: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to reset database: {e}")
    
    # 2. Fetch Gist metadata to get file URLs
    logger.info(f"Fetching Gist metadata for ID: {gist_id}...")
    api_url = f"https://api.github.com/gists/{gist_id}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        gist_data = response.json()
        files = gist_data.get('files', {})
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch Gist metadata: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch Gist metadata: {e}")

    papers_to_ingest = []
    if not files:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No files found in the specified Gist.")

    # 3. Process and embed each file in the Gist
    for filename, file_info in files.items():
        raw_url = file_info.get('raw_url')
        if not raw_url:
            logger.warning(f"File '{filename}' in Gist has no raw_url. Skipping.")
            continue
        
        logger.info(f"Fetching content for file: '{filename}' from URL: {raw_url}")
        try:
            content_response = requests.get(raw_url)
            content_response.raise_for_status()
            content = content_response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch content from {raw_url}: {e}")
            continue

        # Use the whole file content for now
        text_to_embed = content

        embedding = embedding_model.encode(text_to_embed).tolist()
        
        paper_entry = Paper(
            title=filename,
            abstract=content,
            authors="Gist User",
            url=raw_url,
            embedding=embedding
        )
        papers_to_ingest.append(paper_entry)

    # 4. Store all data in the database
    logger.info(f"Ingesting {len(papers_to_ingest)} documents into the database...")
    try:
        db.bulk_save_objects(papers_to_ingest)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database ingestion failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database ingestion failed: {e}")

    return {"message": f"Successfully loaded and ingested {len(papers_to_ingest)} documents.", "documents_ingested": len(papers_to_ingest)}
