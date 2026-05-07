"""
main.py — FastAPI application.

Key changes from v2:
    - POST /upload returns immediately with a job_id.
      Ingestion runs in a FastAPI BackgroundTask — the UI never freezes.
    - GET /ingest/status/{job_id} — frontend polls this to track progress.
    - VectorFactory used everywhere — no direct Chroma imports.
    - In-memory job store (dict) tracks ingestion state.
      For multi-worker production deployments, replace with Redis:
          import redis; r = redis.Redis(); r.hset(job_id, ...)
"""
import os
import uuid
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from .models import BookRecord # Ensure this is imported for type annotations
from .logger import get_logger
from .config_loader import general_settings

logger = get_logger(
    level=general_settings["app"]["log_level"],
    log_file=general_settings["app"]["log_file"].split("/")[-1],
)

from .models import (
    QuestionRequest, AnswerResponse,
    IngestResponse, BooksResponse,
)
from .database_pool import (
    setup_db, log_ingestion, list_books, close_all_connections,
)
from .encryption import get_encryption
from .ingest_pdf import ingest, load_existing
from .query import bot

load_dotenv()

VECTOR_DB_DIR   = general_settings["embedding"]["vector_db_dir"]
UPLOADS_DIR     = general_settings["ingestion"]["uploads_dir"]
EMBEDDING_MODEL = general_settings["embedding"]["model_name"]
APP_NAME        = general_settings["app"]["name"]
APP_VERSION     = general_settings["app"]["version"]


# ── In-memory job store ───────────────────────────────────────────────────────
# Tracks background ingestion jobs.
# Structure: { job_id: { status, filename, chunks, error } }
#
# Status values:
#   "queued"     — job received, not started yet
#   "processing" — ingestion running in background
#   "complete"   — done, vector store updated
#   "failed"     — ingestion raised an exception (see error field)
_jobs: dict[str, dict] = {}


def _get_job(job_id: str) -> dict:
    """Return job record or raise 404."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


# ── Background task ───────────────────────────────────────────────────────────
def _run_ingest(
    job_id:     str,
    pdf_path:   str,
    filename:   str,
    app_state,             # This should be FastAPI's app.state object
):
    """
    The actual ingestion work. Runs in a background thread via FastAPI's
    BackgroundTasks — the HTTP response is already sent before this starts.

    Args:
        job_id: Unique identifier for the ingestion job.
        pdf_path: Path to the uploaded PDF file.
        filename: Name of the uploaded file.
        app_state: FastAPI app.state object (i.e., request.app.state) — used to update vector_db.

    Why not async?
        Langchain's PDF loader and ChromaDB writes are synchronous blocking
        operations. Running them in an async function would block the event
        loop. FastAPI BackgroundTasks run sync functions in a thread pool.
    """
    _jobs[job_id]["status"] = "processing"
    logger.info(f"BG_INGEST | job={job_id} | Starting: {filename}")

    try:
        # 1. Ingest the new PDF into the PERSISTENT directory
        # Ensure your ingest() function is configured to append to the same 'vector_db' folder
        vector_store, chunk_count = ingest(pdf_path=pdf_path)

        # 2. Re-load the GLOBAL store that now contains BOTH PDFs
        # Use your factory to ensure it pulls the combined index
        from .ingest_pdf import load_existing
        updated_full_store = load_existing(app_state.embeddings)

        # 3. Hot-swap to the FULL combined store
        if updated_full_store:
            app_state.vector_db = updated_full_store
        # Hot-swap the vector store so /ask uses new knowledge immediately
        #app_state.vector_db = vector_store

        log_ingestion(filename, chunk_count)

        _jobs[job_id].update({
            "status":  "complete",
            "chunks":  chunk_count,
            "error":   None,
        })
        logger.info(
            f"BG_INGEST | job={job_id} | Complete | "
            f"file={filename} | chunks={chunk_count}"
        )

    except Exception as e:
        _jobs[job_id].update({
            "status": "failed",
            "error":  str(e),
        })
        logger.error(f"BG_INGEST | job={job_id} | FAILED: {e}")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"STARTUP | {APP_NAME} v{APP_VERSION}")

    logger.info("STARTUP | Setting up database")
    setup_db()

    logger.info("STARTUP | Initialising encryption")
    get_encryption()

    logger.info(f"STARTUP | Loading embedding model: {EMBEDDING_MODEL}")
    app.state.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logger.info("STARTUP | Embeddings ready")

    logger.info("STARTUP | Loading vector store")
    app.state.vector_db = load_existing(app.state.embeddings)
    if app.state.vector_db:
        logger.info("STARTUP | Vector store loaded — ready")
    else:
        logger.warning("STARTUP | No vector store — upload a PDF first")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.critical("STARTUP | GROQ_API_KEY missing")
        raise RuntimeError("GROQ_API_KEY not set")
    app.state.groq_client = Groq(api_key=api_key)
    logger.info("STARTUP | Groq client ready")

    Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"STARTUP | Ready — uploads dir: {UPLOADS_DIR}")
    yield

    logger.info("SHUTDOWN | Closing DB pool")
    close_all_connections()
    logger.info("SHUTDOWN | Done")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_NAME,
    description="Modular agentic RAG — plug and play for any document corpus",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dependencies ──────────────────────────────────────────────────────────────
def get_vector_db(request: Request):
    if request.app.state.vector_db is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not ready. Upload a PDF first.",
        )
    return request.app.state.vector_db


def get_groq_client(request: Request):
    return request.app.state.groq_client


def get_embeddings(request: Request):
    return request.app.state.embeddings


# ── Response model for job status ─────────────────────────────────────────────
class JobStatus(BaseModel):
    job_id:   str
    status:   str           # queued | processing | complete | failed
    filename: str
    chunks:   int  = 0
    error:    str | None = None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "live", "service": APP_NAME, "docs": "/docs"}


@app.get("/health")
def health(request: Request):
    return {
        "database":     "connected",
        "vector_store": "loaded" if request.app.state.vector_db else "not loaded",
        "groq":         "connected",
        "encryption":   "active",
    }


@app.post("/upload", response_model=JobStatus, status_code=202)
async def upload_pdf(
    request:          Request,
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(...),
):
    """
    Upload a PDF and start background ingestion.

    Returns HTTP 202 Accepted immediately with a job_id.
    The client polls GET /ingest/status/{job_id} to track progress.

    Why 202 and not 200?
        202 Accepted is the correct HTTP status for "I received your
        request and will process it, but I haven't finished yet."
        200 would imply the work is done, which it isn't.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")

    # Save file synchronously — this is fast (just disk write)
    save_path = Path(UPLOADS_DIR) / file.filename
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    size_kb = save_path.stat().st_size // 1024
    logger.info(f"UPLOAD | Saved {file.filename} ({size_kb} KB)")

    # Register job
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status":   "queued",
        "filename": file.filename,
        "chunks":   0,
        "error":    None,
    }
    logger.info(f"UPLOAD | Queued job {job_id} for {file.filename}")

    # Schedule background ingestion — returns immediately
    background_tasks.add_task(
        _run_ingest,
        job_id=job_id,
        pdf_path=str(save_path),
        filename=file.filename,
        app_state=request.app.state,
    )

    return JobStatus(job_id=job_id, status="queued", filename=file.filename)


@app.get("/ingest/status/{job_id}", response_model=JobStatus)
def ingest_status(job_id: str):
    """
    Poll ingestion job status.

    Frontend polls this every 2 seconds after POST /upload.
    When status == "complete", the knowledge base is updated.
    When status == "failed", display the error field to the user.
    """
    job = _get_job(job_id)
    return JobStatus(job_id=job_id, **job)


@app.get("/ingest/jobs")
def list_jobs():
    """List all ingestion jobs (useful for debugging)."""
    return {"jobs": [{"job_id": jid, **data} for jid, data in _jobs.items()]}




@app.get("/books", response_model=BooksResponse)
async def books():  # Added async
    logger.debug("ROUTE | GET /books")
    
    # Await the coroutine to get the actual data
    data = await list_books() 
    
    # Map the dictionaries to BookRecord objects to satisfy Pydantic
    book_records = [BookRecord(**b) for b in data]
    
    return BooksResponse(books=book_records)

@app.post("/ask", response_model=AnswerResponse)
async def ask( # 1. Add 'async' here
    body:       QuestionRequest,
    vector_db=  Depends(get_vector_db),
    groq_client=Depends(get_groq_client),
):
    logger.event("ask", session=body.session_id, q=body.question[:60])

    try:
        # 2. Add 'await' here to resolve the dictionary
        result = await bot(
            question=body.question,
            vector_db=vector_db,
            client=groq_client,
            session_id=body.session_id,
            fmt=body.format,
        )
    except Exception as e:
        logger.error(f"ROUTE | /ask failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Now result["answer"] will work because result is a dict, not a Coroutine
    return AnswerResponse(
        answer=result["answer"],
        session_id=body.session_id,
        intent=result["intent"],
        tool_used=result["tool_used"],
    )
