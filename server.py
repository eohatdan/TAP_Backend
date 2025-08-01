# server.py

import os
import requests
import uuid
import logging
from typing import List

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import create_engine, text, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import OperationalError

# A key dependency for vector embeddings
from pgvector.sqlalchemy import Vector

# --- RAG Specific Dependencies ---
# We'll use a small, fast, open-source embedding model for prototyping.
# The 'all-MiniLM-L6-v2' model produces 384-dimensional embeddings.
from sentence_transformers import SentenceTransformer

# --- Logging Configuration ---
# Set up basic logging for better visibility into the server's state
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
    # In a production app, you would raise an exception here to prevent startup
    # For this prototype, we'll let the startup event handle the connection failure
    pass

# SQLAlchemy setup
# We use a declarative base for defining our database tables as Python classes
Base = declarative_base()

# Create a database engine
# `pool_pre_ping=True` helps maintain a stable connection pool on Render.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Create a session local class for database interactions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Database Models ---
# This class defines the structure of our 'papers' table.
# It includes a vector column for the embedding.
class Paper(Base):
    __tablename__ = 'papers'
    
    # We use a UUID as the primary key for uniqueness
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String)
    abstract = Column(String)
    authors = Column(String)
    url = Column(String)
    
    # The crucial pgvector column for our embeddings
    embedding = Column(Vector(384))

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

@app.on_event("startup")
async def startup_event():
    """
    This event handler runs once when the application starts up.
    It's used to:
    1. Verify a successful database connection.
    2. Load the embedding model into memory for fast access.
    3. Ensure the 'papers' table exists in the database.
    """
    logger.info("Backend application starting up...")
    
    # 1. Verify database connection
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Successfully connected to the database!")
    except OperationalError as e:
        logger.error(f"Database connection failed on startup: {e}")
        # Raising an exception here will prevent the application from starting
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")

    # 2. Load the embedding model
    global embedding_model
    try:
        logger.info("Loading Sentence-Transformer model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # If the model fails to load, the app is non-functional for ingestion/search
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
        # A simple query to check the database connection
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

    # 1. Fetch data from Semantic Scholar Public API
    logger.info(f"Fetching papers for query: '{query}' with limit={limit}...")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={limit}&fields=title,abstract,authors,url"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for 4xx/5xx status codes
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch data from Semantic Scholar: {e}")

    papers_to_ingest = []
    if 'data' not in data:
        return {"message": "No papers found for this query."}

    # 2. Process and embed each paper
    for paper_data in data['data']:
        if paper_data.get('abstract') and paper_data.get('title'):
            authors_list = [author['name'] for author in paper_data.get('authors', [])]
            
            # Combine title and abstract for a richer embedding
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

    # 3. Store in the database
    logger.info(f"Ingesting {len(papers_to_ingest)} papers into the database...")
    try:
        db.bulk_save_objects(papers_to_ingest)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database ingestion failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database ingestion failed: {e}")
    
    return {"message": f"Successfully ingested {len(papers_to_ingest)} papers.", "papers_ingested": len(papers_to_ingest)}

# Assuming these imports are already present in your server.py
# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# from typing import List

# Placeholder for your LLM client (e.g., OpenAI, Gemini, etc.)
# from your_llm_library import YourLLMClient
# llm_client = YourLLMClient()

# Placeholder for your Vector Database client
# from your_vector_db_library import YourVectorDBClient
# vector_db_client = YourVectorDBClient()

# This endpoint is a GET request and takes a single query parameter.
@app.get("/query")
async def query_data(query: str):
    """
    Performs a RAG query against the ingested data.
    """
    if not query:
        return {"error": "Query parameter is required."}, 400

    try:
        # Step 1: Search the vector database for relevant documents
        # This is a placeholder. You need to implement the actual search logic here.
        # The search should return chunks of text that are most similar to the user's query.
        relevant_documents = await vector_db_client.search_vectors(query, top_k=5)

        # Step 2: Combine the user's query with the retrieved documents into a single prompt
        augmented_prompt = (
            "Based on the following documents, answer the user's question.\n\n"
            "Documents:\n"
            "```\n"
            f"{' '.join(relevant_documents)}\n"
            "```\n\n"
            "User's Question:\n"
            f"{query}"
        )

        # Step 3: Send the augmented prompt to the LLM for a final response
        # This is a placeholder. You need to implement the LLM API call here.
        llm_response = await llm_client.generate_response(augmented_prompt)

        return {"response": llm_response}

    except Exception as e:
        # Handle any errors that occur during the process
        print(f"Error during query processing: {e}")
        return {"error": f"An internal server error occurred: {e}"}, 500
