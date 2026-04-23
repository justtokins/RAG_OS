"""
PDF ingestion pipeline.

Two public functions:
    ingest()        — load, chunk, embed, persist to ChromaDB
    load_existing() — load an already-built ChromaDB store

The embedding model is passed in (not created here) because:
    - It takes ~5 seconds and 90MB to load
    - main.py creates it once at startup and passes it everywhere
    - Creating it inside ingest() would reload it on every upload
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config_loader import general_settings
from logger import get_logger

logger = get_logger()

CHUNK_SIZE    = general_settings["ingestion"]["chunk_size"]
CHUNK_OVERLAP = general_settings["ingestion"]["chunk_overlap"]
SEPARATORS    = general_settings["ingestion"]["separators"]


def ingest(
    pdf_path:    str,
    persist_dir: str,
    embeddings:  HuggingFaceEmbeddings,
) -> tuple[Chroma, int]:
    """
    Load a PDF, chunk it, embed it, and persist to ChromaDB.

    Parameters
    ----------
    pdf_path    : absolute or relative path to the PDF file
    persist_dir : directory where ChromaDB will be saved / updated
    embeddings  : pre-loaded HuggingFaceEmbeddings instance

    Returns
    -------
    (vector_db, chunk_count)

    Why chunk_size=1000, overlap=200?
        1000 characters fits one full concept explanation without
        cutting it off. 200 character overlap means a concept that
        spans a page boundary is not split in half — the end of one
        chunk appears at the start of the next, so retrieval never
        misses a concept that straddles a boundary.
    """
    path = Path(pdf_path)
    if not path.exists():
        logger.error(f"INGEST | PDF not found: {pdf_path}")
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"INGEST | Loading PDF: {path.name} ({path.stat().st_size // 1024} KB)")

    loader   = PyPDFLoader(str(path))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = loader.load_and_split(text_splitter=splitter)
    logger.info(f"INGEST | {len(chunks)} chunks created from {path.name}")

    logger.info(f"INGEST | Embedding and storing in {persist_dir} ...")
    try:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
    except Exception as e:
        logger.error(f"INGEST | ChromaDB storage failed: {e}")
        raise

    logger.info(f"INGEST | Complete — {len(chunks)} chunks stored in {persist_dir}")
    return vector_db, len(chunks)


def load_existing(
    persist_dir: str,
    embeddings:  HuggingFaceEmbeddings,
) -> Chroma | None:
    """
    Load an existing ChromaDB vector store from disk.
    Returns None if the store has not been built yet —
    caller is responsible for handling the None case.
    """
    if not Path(persist_dir).exists():
        logger.warning(f"INGEST | No vector store found at {persist_dir}")
        return None

    logger.info(f"INGEST | Loading existing vector store from {persist_dir}")
    try:
        store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        logger.info("INGEST | Vector store loaded successfully")
        return store
    except Exception as e:
        logger.error(f"INGEST | Failed to load vector store: {e}")
        raise