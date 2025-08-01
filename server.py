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

# Assuming these imports are already present in your server.py
# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# from typing import List
import requests # We need to import requests for this
from typing import Optional

# Placeholder for your SentenceTransformer model
# from sentence_transformers import SentenceTransformer
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Placeholder for your Vector Database client
# from your_vector_db_library import YourVectorDBClient
# vector_db_client = YourVectorDBClient()

# Define the request body schema for data ingestion
class IngestDataRequest(BaseModel):
    # 'text_to_ingest' is optional, as we can get text from a URL
    text_to_ingest: Optional[str] = None
    # 'url' is also optional, but one of the two must be provided
    url: Optional[str] = "https://storage.googleapis.com/file_xfer_bucket/test_data.txt"

@app.post("/ingest-data")
async def ingest_data(request_body: IngestDataRequest):
    """
    Ingests text data from either the request body or a URL, chunks it,
    and stores its embeddings in the vector database.
    """
    try:
        # Check if either text or a URL was provided
        if not request_body.text_to_ingest and not request_body.url:
            return {"error": "Either 'text_to_ingest' or 'url' must be provided."}, 400
        
        text_content = ""
        # If a URL is provided, fetch the text from it
        if request_body.url:
            print(f"Fetching data from URL: {request_body.url}")
            response = requests.get(request_body.url)
            response.raise_for_status() # Raise an exception for bad status codes
            text_content = response.text
        # Otherwise, use the text from the request body
        else:
            text_content = request_body.text_to_ingest

        # Step 1: Chunk the text content
        # This is a simple placeholder. For real applications, you'd use a more
        # sophisticated chunking strategy (e.g., based on sentences or paragraphs).
        # We'll split by double newline here as a simple example.
        text_chunks = text_content.split("\n\n")

        # Step 2: Create embeddings for each chunk
        # This is a placeholder. You need to implement the actual embedding logic here.
        # embeddings = embedding_model.encode(text_chunks)
        embeddings = [embedding_model.encode(chunk) for chunk in text_chunks]

        # Step 3: Store the chunks and embeddings in the vector database
        # This is a placeholder. You need to implement the actual storage logic here.
        await vector_db_client.store_embeddings_and_texts(embeddings, text_chunks)

        return {"message": f"Successfully ingested {len(text_chunks)} chunks of data."}

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from URL: {e}")
        return {"error": f"An error occurred while fetching data: {e}"}, 500
    except Exception as e:
        # Handle any other errors that occur during the ingestion process
        print(f"Error during data ingestion: {e}")
        return {"error": f"An internal server error occurred: {e}"}, 500


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
