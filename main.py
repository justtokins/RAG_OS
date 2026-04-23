import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings

from config_loader import general_settings
from models import QuestionRequest, AnswerResponse, IngestResponse, HomeResponse, HealthResponse, BooksResponse
from logger import get_logger
from database_pool import setup_db, log_ingestion, list_books, close_all_connections
from ingest_pdf import ingest, load_existing
from query import bot

# Initialize logger
logger = get_logger(level=general_settings['app']['log_level'])

load_dotenv()

VECTOR_DB_DIR  = general_settings['embedding']['vector_db_dir']
UPLOADS_DIR    = general_settings['ingestion']['uploads_dir']
EMBEDDING_MODEL = general_settings['embedding']['model_name']
APP_TITLE      = general_settings['app']['name']
APP_VERSION    = general_settings['app']['version']


#Startup & Shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Everything in this block runs ONCE when the server starts.
    Resources are stored in app.state and shared across all requests.
    Nothing is initialised per-request — that would be extremely slow.

    Startup order:
        1. PostgreSQL tables created
        2. Embedding model loaded (90MB, takes ~5 seconds)
        3. Vector store loaded from disk (if it exists)
        4. Groq client initialised
        5. Upload folder created

    On shutdown (after yield): cleanup happens here if needed.
    """
    print("[startup] Setting up database...")
    setup_db()

    print("[startup] Loading embedding model (this takes ~5 seconds)...")
    app.state.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("[startup] Loading vector store...")
    app.state.vector_db = load_existing(VECTOR_DB_DIR, app.state.embeddings)
    if app.state.vector_db:
        print("[startup] Vector store loaded — system ready to answer questions")
    else:
        print("[startup] No vector store found — upload and ingest a PDF first")

    print("[startup] Connecting to Groq...")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing from environment")
    app.state.groq_client = Groq(api_key=api_key)

    Path(UPLOADS_DIR).mkdir(exist_ok=True)
    print("[startup] Ready.")

    yield  # server runs here

    print("[shutdown] Cleaning up...")
    close_all_connections()


#App & Middleware 
app = FastAPI(
    title=APP_TITLE,
    description="Agentic RAG system for Operating Systems textbooks",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production to your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)


#Dependency: inject shared resources into route handlers
def get_vector_db(request: Request):
    """
    FastAPI dependency — injects vector_db from app.state into routes.
    Raises 503 if no PDF has been ingested yet.
    """
    if request.app.state.vector_db is None:
        raise HTTPException(
            status_code=503,
            detail="No knowledge base found. Upload and ingest a PDF first."
        )
    return request.app.state.vector_db


def get_groq_client(request: Request):
    return request.app.state.groq_client


def get_embeddings(request: Request):
    return request.app.state.embeddings


#Routes
@app.get("/")
def home():
    return {
        "status":  "live",
        "service": "OS RAG Assistant",
        "docs":    "/docs",
    }


@app.get("/health")
def health(request: Request):
    """
    Health check — tells you whether the system is fully ready.
    Call this after deployment to confirm everything started correctly.
    """
    return {
        "database":     "connected",
        "vector_store": "loaded" if request.app.state.vector_db else "not loaded",
        "groq":         "connected" if request.app.state.groq_client else "error",
    }


@app.post("/upload")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    embeddings=Depends(get_embeddings),
):
    """
    Upload a PDF to the server.

    This saves the file to disk and immediately ingests it into
    the vector store. After this call the system can answer
    questions about the uploaded book.

    Accepts multipart/form-data — any HTTP client or frontend
    can call this with a file picker input.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no name.")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")
    
    filename = file.filename.rsplit(".", 1)[0]  # remove .pdf extension for logging

    # Save to uploads folder
    save_path = Path(UPLOADS_DIR) / file.filename
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    print(f"[upload] Saved {file.filename} ({save_path.stat().st_size // 1024} KB)")

    # Ingest immediately
    try:
        vector_db, chunk_count = ingest(
            pdf_path=str(save_path),
            persist_dir=VECTOR_DB_DIR,
            embeddings=embeddings,
        )
        # Update the shared vector_db so /ask works immediately
        request.app.state.vector_db = vector_db

        log_ingestion(file.filename, chunk_count)

        return IngestResponse(
            message=f"Successfully ingested {file.filename}",
            filename=file.filename,
            chunks=chunk_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.get("/books")
def books():
    """List all books that have been ingested into the knowledge base."""
    return {"books": list_books()}


@app.post("/ask", response_model=AnswerResponse)
def ask(
    body: QuestionRequest,
    vector_db=Depends(get_vector_db),
    groq_client=Depends(get_groq_client),
):
    """
    Ask a question. The agent routes it to the right tool,
    retrieves context, and the LLM generates a clear answer.
    """
    answer = bot(
        question=body.question,
        vector_db=vector_db,
        client=groq_client,
        session_id=body.session_id,
    )
    return AnswerResponse(answer=answer, session_id=body.session_id)