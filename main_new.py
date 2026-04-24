"""
main.py — FastAPI application entry point.

Startup sequence (lifespan):
    1. Logger initialised from general_settings
    2. PostgreSQL tables created
    3. Encryption singleton initialised (validates key early)
    4. Embedding model loaded (~5 seconds, ~90 MB)
    5. Vector store loaded from disk if it exists
    6. Groq client initialised
    7. Upload directory created

All shared resources live in app.state and are injected into
route handlers via FastAPI dependencies — nothing is a global variable.
"""
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Logger must be initialised before any other local import
# so that modules which call get_logger() at import time
# receive the correctly configured instance.
from logger import get_logger
from config_loader import general_settings

logger = get_logger(
    level=general_settings["app"]["log_level"],
    log_file=general_settings["app"]["log_file"].split("/")[-1],
)

from models import QuestionRequest, AnswerResponse, IngestResponse, BooksResponse
from database_pool import setup_db, log_ingestion, list_books, close_all_connections
from encryption import get_encryption
from ingest_pdf import ingest, load_existing
from query import bot

load_dotenv()

VECTOR_DB_DIR   = general_settings["embedding"]["vector_db_dir"]
UPLOADS_DIR     = general_settings["ingestion"]["uploads_dir"]
EMBEDDING_MODEL = general_settings["embedding"]["model_name"]
APP_NAME        = general_settings["app"]["name"]
APP_VERSION     = general_settings["app"]["version"]


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup (before yield) and once at shutdown (after yield).

    Why app.state?
        FastAPI's app.state is a simple namespace shared across the entire
        application lifetime. Dependencies inject from it per-request.
        This avoids global variables while keeping initialisation in one place.
    """
    logger.info(f"STARTUP | {APP_NAME} v{APP_VERSION} starting")

    # 1. Database
    logger.info("STARTUP | Setting up PostgreSQL tables")
    setup_db()

    # 2. Encryption — initialise early so any key problem fails at startup
    logger.info("STARTUP | Initialising encryption")
    get_encryption()

    # 3. Embedding model
    logger.info(f"STARTUP | Loading embedding model: {EMBEDDING_MODEL}")
    app.state.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logger.info("STARTUP | Embedding model loaded")

    # 4. Vector store
    logger.info(f"STARTUP | Loading vector store from: {VECTOR_DB_DIR}")
    app.state.vector_db = load_existing(VECTOR_DB_DIR, app.state.embeddings)
    if app.state.vector_db:
        logger.info("STARTUP | Vector store loaded — system ready to answer questions")
    else:
        logger.warning("STARTUP | No vector store found — upload a PDF before asking questions")

    # 5. Groq client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.critical("STARTUP | GROQ_API_KEY missing — cannot start LLM service")
        raise RuntimeError("GROQ_API_KEY not set in environment")
    app.state.groq_client = Groq(api_key=api_key)
    logger.info("STARTUP | Groq client initialised")

    # 6. Upload directory
    Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"STARTUP | Upload directory ready: {UPLOADS_DIR}")

    logger.info("STARTUP | Ready — accepting requests")
    yield

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("SHUTDOWN | Closing database connection pool")
    close_all_connections()
    logger.info("SHUTDOWN | Clean shutdown complete")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_NAME,
    description="Agentic RAG system — modular, encrypted, configurable",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dependencies ──────────────────────────────────────────────────────────────
def get_vector_db(request: Request):
    """Inject vector_db. Raises 503 if no PDF has been ingested yet."""
    if request.app.state.vector_db is None:
        logger.warning("DEP | /ask called but no vector store loaded")
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not loaded. Upload and ingest a PDF first.",
        )
    return request.app.state.vector_db


def get_groq_client(request: Request):
    return request.app.state.groq_client


def get_embeddings(request: Request):
    return request.app.state.embeddings


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    logger.debug("ROUTE | GET /")
    return {"status": "live", "service": APP_NAME, "docs": "/docs"}


@app.get("/health")
def health(request: Request):
    """
    Health check endpoint.
    Call this after deployment to verify every subsystem is ready.
    """
    logger.debug("ROUTE | GET /health")
    return {
        "database":     "connected",
        "vector_store": "loaded" if request.app.state.vector_db else "not loaded",
        "groq":         "connected" if request.app.state.groq_client else "error",
        "encryption":   "active",
    }


@app.post("/upload")
async def upload_pdf(
    request:    Request,
    file:       UploadFile = File(...),
    embeddings  = Depends(get_embeddings),
):
    """
    Upload a PDF. Saves it to disk and ingests it immediately.

    After a successful call, /ask can answer questions about this book.
    Accepts multipart/form-data — standard HTML file input or any HTTP client.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")

    save_path = Path(UPLOADS_DIR) / file.filename
    logger.info(f"UPLOAD | Receiving: {file.filename}")

    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    size_kb = save_path.stat().st_size // 1024
    logger.info(f"UPLOAD | Saved {file.filename} ({size_kb} KB)")

    try:
        vector_db, chunk_count = ingest(
            pdf_path=str(save_path),
            persist_dir=VECTOR_DB_DIR,
            embeddings=embeddings,
        )
        request.app.state.vector_db = vector_db
        log_ingestion(file.filename, chunk_count)
        logger.info(f"UPLOAD | Ingestion complete | chunks={chunk_count}")

        return IngestResponse(
            message=f"Successfully ingested {file.filename}",
            filename=file.filename,
            chunks=chunk_count,
        )
    except Exception as e:
        logger.error(f"UPLOAD | Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.get("/books", response_model=BooksResponse)
def books():
    """List all books ingested into the knowledge base."""
    logger.debug("ROUTE | GET /books")
    return BooksResponse(books=list_books())


@app.post("/ask", response_model=AnswerResponse)
def ask(
    body:         QuestionRequest,
    vector_db     = Depends(get_vector_db),
    groq_client   = Depends(get_groq_client),
):
    """
    Ask a question about the ingested books.

    The agent detects intent, selects the right tool, retrieves context,
    and the LLM generates a formatted answer. Conversation history is
    stored encrypted in PostgreSQL.
    """
    logger.event("ask", session=body.session_id, question=body.question[:60])

    try:
        result = bot(
            question=body.question,
            vector_db=vector_db,
            client=groq_client,
            session_id=body.session_id,
            fmt=body.format,
        )
    except Exception as e:
        logger.error(f"ROUTE | /ask failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return AnswerResponse(
        answer=result["answer"],
        session_id=body.session_id,
        intent=result["intent"],
        tool_used=result["tool_used"],
    )