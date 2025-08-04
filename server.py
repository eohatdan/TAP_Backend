from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
import sys
import re
import psycopg2
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector
from psycopg2.extras import Json
import asyncio
from concurrent.futures iimport logging
import uuid
import logging
from typing import List, Dict
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import create_engine, text, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import OperationalError
# A key dependency for vector embeddings
from pgvector.sqlalchemy import Vector




# Initialize FastAPI application
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# --- CORS Middleware Configuration ---
# Your GitHub Pages URL is the origin for the frontend
origins = [
    "https://eohatdan.github.io",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- End CORS Middleware ---

# --- Configuration from Environment Variables ---
DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- RAG Components Initialization ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
else:
    gemini_model = None
    logging.warning("GEMINI_API_KEY not found. Gemini API will not be available.")

executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)

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

def create_table_if_not_exists():
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            logging.warning("Skipping table creation due to failed database connection.")
            return
        cur = conn.cursor()
        
        # The DROP TABLE command has been removed to allow the database to persist data.
        # cur.execute("DROP TABLE IF EXISTS documents;")
        # conn.commit()
        # logging.info("Documents table dropped for a clean restart.")
        
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

create_table_if_not_exists()

def get_raw_text_from_google_drive(url):
    match = re.search(r'file/d/([^/]+)', url)
    if match:
        file_id = match.group(1)
        docs_export_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        return docs_export_url
    return url

class IngestDataRequest(BaseModel):
    text_to_ingest: Optional[str] = None
    url: Optional[str] = None

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the TAP backend."}

@app.post("/ingest-data")
async def ingest_data(request_body: IngestDataRequest):
    try:
        if not request_body.text_to_ingest and not request_body.url:
            return {"error": "Either 'text_to_ingest' or 'url' must be provided."}, 400
        text_content = ""
        if request_body.url:
            logging.info(f"Fetching data from URL: {request_body.url}")
            processed_url = get_raw_text_from_google_drive(request_body.url)
            response = requests.get(processed_url)
            response.raise_for_status()
            text_content = response.text
        else:
            text_content = request_body.text_to_ingest
        text_chunks = text_content.split("\n\n")
        logging.info(f"Chunked data into {len(text_chunks)} chunks.")
        embeddings = embedding_model.encode(text_chunks)
        logging.info(f"Created {len(embeddings)} embeddings.")
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

@app.post("/query")
async def query_data(request_body: QueryRequest):
    try:
        if not request_body.query:
            return {"error": "Query parameter is required in the body."}, 400
        query_embedding = embedding_model.encode([request_body.query])[0]
        query_embedding_list = query_embedding.tolist()
        logging.info(f"Query embedding list generated. Length: {len(query_embedding_list)}")
        relevant_documents = await asyncio.get_event_loop().run_in_executor(
            executor, search_db_for_vectors, query_embedding_list
        )
        logging.info(f"Found {len(relevant_documents)} relevant documents from database.")
        logging.info(f"Retrieved documents: {relevant_documents}")
        if not relevant_documents:
            return {"response": "I could not find any relevant documents to answer your question."}
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
        logging.info(f"Augmented prompt sent to LLM: {augmented_prompt}")
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
            continue # Skip this file and try the next one

        # A simple way to chunk the text: just use the whole file content for now
        # For a real-world app, you would chunk it here.
        text_to_embed = content

        embedding = embedding_model.encode(text_to_embed).tolist()
        
        paper_entry = Paper(
            title=filename,  # Using filename as title for simplicity
            abstract=content, # Using content as abstract
            authors="Gist User", # Placeholder
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
