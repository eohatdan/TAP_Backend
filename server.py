# Assuming these imports are already present in your server.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
import sys
import re

# --- New Imports for RAG Implementation ---
import psycopg2
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector
from psycopg2.extras import Json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Initialize FastAPI application
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# --- Configuration from Environment Variables ---
DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- RAG Components Initialization ---
# Initialize Sentence Transformer for creating embeddings
# This can be a memory-intensive step. It's done once on startup.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Gemini client
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
else:
    gemini_model = None
    logging.warning("GEMINI_API_KEY not found. Gemini API will not be available.")

# Thread pool for asynchronous database operations
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)

# Helper function to get a database connection
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        logging.info("Successfully connected to the database.")
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if not cur.fetchone():
                logging.error("pgvector extension not installed in the database.")
                conn.close()
                return None
        register_vector(conn)
        return conn
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
        if conn:
            conn.close()
        return None

# Helper function to create the table if it doesn't exist
def create_table_if_not_exists():
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            logging.warning("Skipping table creation due to failed database connection.")
            return

        cur = conn.cursor()
        # Drop table for clean state in debugging
        cur.execute("DROP TABLE IF EXISTS documents;")
        conn.commit()
        logging.info("Documents table dropped for a clean restart.")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(384) NOT NULL
            );
        """)
        conn.commit()
        logging.info("Documents table checked/created successfully.")
    except Exception as e:
        logging.error(f"Error creating table: {e}")
    finally:
        if conn:
            conn.close()

# Create table on startup
create_table_if_not_exists()

# Define the request body schemas
class IngestDataRequest(BaseModel):
    text_to_ingest: Optional[str] = None
    url: Optional[str] = None

class QueryRequest(BaseModel):
    query: str

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the TAP backend."}

# The /ingest-data endpoint for POST requests
@app.post("/ingest-data")
async def ingest_data(request_body: IngestDataRequest):
    """
    Ingests text data from either the request body or a URL, chunks it,
    and stores its embeddings in the vector database.
    """
    try:
        if not request_body.text_to_ingest and not request_body.url:
            return {"error": "Either 'text_to_ingest' or 'url' must be provided."}, 400
        
        text_content = ""
        if request_body.url:
            logging.info(f"Fetching data from URL: {request_body.url}")
            match = re.search(r'document/d/([^/]+)', request_body.url)
            
            if match:
                file_id = match.group(1)
                docs_export_url = f"https://docs.google.com/document/d/{file_id}/export?format=txt"
                response = requests.get(docs_export_url)
            else:
                response = requests.get(request_body.url)
            
            response.raise_for_status()
            text_content = response.text
        else:
            text_content = request_body.text_to_ingest

        # Step 1: Chunk the text content
        text_chunks = text_content.split("\n\n")
        logging.info(f"Chunked data into {len(text_chunks)} chunks.")

        # Step 2: Create embeddings for each chunk
        embeddings = embedding_model.encode(text_chunks)
        logging.info(f"Created {len(embeddings)} embeddings.")

        # Step 3: Store the chunks and embeddings in the vector database
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, insert_chunks_to_db, embeddings, text_chunks)

        return {"message": f"Successfully ingested {len(text_chunks)} chunks of data."}

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from URL: {e}")
        return {"error": f"An error occurred while fetching data: {e}"}, 500
    except Exception as e:
        logging.error(f"Error during data ingestion: {e}")
        return {"error": f"An internal server error occurred: {e}"}, 500

def insert_chunks_to_db(embeddings, text_chunks):
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            logging.warning("Skipping insertion due to failed database connection.")
            return

        cur = conn.cursor()
        for text, embedding in zip(text_chunks, embeddings):
            embedding_list = embedding.tolist()
            cur.execute("INSERT INTO documents (content, embedding) VALUES (%s, %s)", (text, Json(embedding_list)))
        conn.commit()
        logging.info(f"Successfully inserted {len(text_chunks)} rows into the documents table.")
    except Exception as e:
        logging.error(f"Error inserting into database: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# The /query endpoint for POST requests
@app.post("/query")
async def query_data(request_body: QueryRequest):
    """
    Performs a RAG query against the ingested data.
    """
    if not request_body.query:
        return {"error": "Query parameter is required in the body."}, 400

    try:
        # Step 1: Create an embedding for the user's query
        query_embedding = embedding_model.encode([request_body.query])[0]
        query_embedding_list = query_embedding.tolist()
        logging.info(f"Query embedding list generated. Length: {len(query_embedding_list)}")

        # Step 2: Search the vector database for relevant documents
        relevant_documents = await asyncio.get_event_loop().run_in_executor(
            executor, search_db_for_vectors, query_embedding_list
        )
        logging.info(f"Found {len(relevant_documents)} relevant documents from database.")
        logging.info(f"Retrieved documents: {relevant_documents}") # Log the retrieved content
        
        if not relevant_documents:
            return {"response": "I could not find any relevant documents to answer your question."}

        # Step 3: Combine the user's query with the retrieved documents
        augmented_prompt = (
            "Based on the following documents, answer the user's question. "
            "If the documents do not contain enough information, state that clearly.\n\n"
            "Documents:\n"
            "```\n"
            f"{' '.join(relevant_documents)}\n"
            "```\n\n"
            "User's Question:\n"
            f"{request_body.query}"
        )
        logging.info(f"Augmented prompt sent to LLM: {augmented_prompt}") # Log the prompt
        
        # Step 4: Send the augmented prompt to the LLM for a final response
        if not gemini_model:
            return {"error": "LLM API not configured. Please check GEMINI_API_KEY."}, 500
            
        gemini_response = gemini_model.generate_content(augmented_prompt)
        llm_response = gemini_response.text

        return {"response": llm_response}

    except Exception as e:
        logging.error(f"Error during query processing: {e}")
        return {"error": f"An internal server error occurred: {e}"}, 500

def search_db_for_vectors(query_embedding_list):
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            logging.warning("Skipping search due to failed database connection.")
            return []

        cur = conn.cursor()
        cur.execute("""
            SELECT content FROM documents
            ORDER BY embedding <-> %s
            LIMIT 5;
        """, (Json(query_embedding_list),))
        
        results = [row[0] for row in cur.fetchall()]
        return results
    except Exception as e:
        logging.error(f"Error searching database: {e}")
        return []
    finally:
        if conn:
            conn.close()

# --- New Debugging Endpoint ---
@app.get("/check-database")
def check_database():
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return {"error": "Failed to connect to the database."}, 500

        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents;")
        row_count = cur.fetchone()[0]
        return {"message": f"Documents table contains {row_count} rows."}
    except Exception as e:
        logging.error(f"Error checking database: {e}")
        return {"error": f"Error checking database: {e}"}, 500
    finally:
        if conn:
            conn.close()
