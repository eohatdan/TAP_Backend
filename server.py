# server.py

import os
import requests
import uuid
import logging
from typing import List, Dict, Optional # Added Optional for BaseModel
from concurrent.futures import ThreadPoolExecutor
import asyncio # Added for async operations with executor

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware # <--- ADDED THIS IMPORT
from pydantic import BaseModel # Added for request body validation
from sqlalchemy import create_engine, text, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import OperationalError

# A key dependency for vector embeddings
from pgvector.sqlalchemy import Vector
from pgvector.psycopg2 import register_vector # For psycopg2 direct connections if needed, but primarily for SQLAlchemy setup
from psycopg2.extras import Json # For passing embeddings as JSON to psycopg2 (if direct connection is used)

# --- RAG Specific Dependencies ---
from sentence_transformers import SentenceTransformer
import google.generativeai as genai # For Gemini LLM integration

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Embedding Model ---
embedding_model = None

# --- Gemini LLM Model ---
gemini_model = None

# --- Thread Pool Executor for Blocking DB Calls ---
# This is used to run synchronous (blocking) database operations in an async FastAPI app
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)

# --- Database Configuration ---
DATABASE_URL = os.environ.get("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Get Gemini API Key

if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set. Please configure it on Render.")
    # In a production app, you might raise an exception here to prevent startup
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
    
    embedding = Column(Vector(384)) # Ensure this matches your embedding model's dimension

# --- Dependency to get a database session ---
def get_db():
    """
    Dependency function to provide a database session to API endpoints.
    It automatically closes the session after the request is complete.
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

# --- CORS Middleware Configuration ---
# Your GitHub Pages URL is the origin for the frontend
origins = [
    "https://eohatdan.github.io", # <--- Make sure this exact URL is here
    # Add other origins if needed, e.g., for local development:
    # "http://localhost",
    # "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)
# --- End CORS Middleware ---


@app.on_event("startup")
async def startup_event():
    """
    This event handler runs once when the application starts up.
    It's used to:
    1. Verify a successful database connection.
    2. Load the embedding model into memory for fast access.
    3. Initialize the Gemini LLM model.
    4. Ensure the 'papers' table exists in the database.
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
        logger.error(f"Error loading embedding model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load embedding model.")

    # 3. Initialize Gemini LLM model
    global gemini_model
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("Gemini LLM model initialized.")
        except Exception as e:
            logger.error(f"Error initializing Gemini LLM: {e}")
            gemini_model = None # Set to None if initialization fails
            logger.warning("Gemini LLM will not be available due to initialization error.")
    else:
        logger.warning("GEMINI_API_KEY not found. Gemini LLM will not be available.")
        gemini_model = None

    # 4. Create the 'papers' table if it doesn't exist
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

# --- Ingestion Endpoints ---
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
        # Use SQLAlchemy's metadata to drop and create the table
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine) # Re-create the table
        logger.info("Existing 'papers' table dropped and re-created.")
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
            title=filename, # Use filename as title for Gist files
            abstract=content, # Use content as abstract
            authors="Gist User", # Placeholder for Gist files
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

# --- RAG Query Endpoint ---
class QueryRequest(BaseModel):
    query: str
    limit: int = 5 # Number of relevant documents to retrieve

@app.post("/query")
async def query_data(request_body: QueryRequest, db: SessionLocal = Depends(get_db)):
    """
    Performs a semantic search on ingested documents and uses an LLM (Gemini)
    to generate a response based on the most relevant documents.
    """
    global embedding_model, gemini_model

    if embedding_model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Embedding model not loaded.")
    if gemini_model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="LLM API not configured or failed to initialize.")

    if not request_body.query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query parameter is required in the body.")

    logger.info(f"Received query: '{request_body.query}'")

    # 1. Generate embedding for the query
    query_embedding = embedding_model.encode([request_body.query])[0].tolist()
    logger.info(f"Query embedding generated. Length: {len(query_embedding)}")

    # 2. Search database for relevant documents (using pgvector)
    # This part needs to run in the executor because SQLAlchemy operations can be blocking
    relevant_papers = await asyncio.get_event_loop().run_in_executor(
        executor, search_db_for_vectors_sync, db, query_embedding, request_body.limit
    )
    
    logger.info(f"Found {len(relevant_papers)} relevant documents from database.")
    
    if not relevant_papers:
        return {"response": "I could not find any relevant documents in the knowledge base to answer your question."}

    # Format relevant documents for the LLM prompt
    documents_text = "\n\n".join([
        f"Title: {p.title}\nAuthors: {p.authors}\nAbstract: {p.abstract}" for p in relevant_papers
    ])

    # 3. Augment prompt and generate response using LLM
    augmented_prompt = (
        "Based on the following documents, answer the user's question. "
        "If the documents do not contain enough information, state that clearly and concisely. "
        "Do not use external knowledge.\n\n"
        "Documents:\n"
        "```\n"
        f"{documents_text}\n"
        "```\n\n"
        "User's Question:\n"
        f"{request_body.query}"
    )
    
    logger.info("Sending augmented prompt to LLM...")
    try:
        gemini_response = gemini_model.generate_content(augmented_prompt)
        llm_response = gemini_response.text
        logger.info("LLM response received.")
        return {"response": llm_response, "relevant_documents": [
            {"title": p.title, "authors": p.authors, "url": p.url, "abstract": p.abstract} for p in relevant_papers
        ]}
    except Exception as e:
        logger.error(f"Error during LLM generation: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM generation failed: {e}")

# Synchronous function to be run in ThreadPoolExecutor
def search_db_for_vectors_sync(db: SessionLocal, query_embedding_list: List[float], limit: int) -> List[Paper]:
    """
    Performs the vector similarity search in the database.
    This function is synchronous and designed to be run in an executor.
    """
    try:
        # Use SQLAlchemy session to query the database
        # The <-> operator is for L2 distance in pgvector
        # You can change to <=> for cosine distance if preferred
        results = db.query(Paper).order_by(
            Paper.embedding.l2_distance(query_embedding_list)
        ).limit(limit).all()
        return results
    except Exception as e:
        logger.error(f"Error searching database for vectors: {e}")
        return []

# --- Database Check Endpoint (already exists as /health, but keeping for reference if needed) ---
# @app.get("/check-database")
# def check_database():
#     # This endpoint is effectively covered by /health now
#     pass
