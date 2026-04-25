"""
ingest_pdf.py — PDF loading, chunking, and vector store ingestion.

VectorFactory replaces every direct Chroma reference.
This file no longer knows or cares which vector store is being used.
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_factory import VectorFactory
from config_loader import general_settings
from logger import get_logger

logger = get_logger()

CHUNK_SIZE    = general_settings["ingestion"]["chunk_size"]
CHUNK_OVERLAP = general_settings["ingestion"]["chunk_overlap"]
SEPARATORS    = general_settings["ingestion"]["separators"]


def ingest(
    pdf_path:   str,
    embeddings: HuggingFaceEmbeddings,
) -> tuple[VectorStore, int]:
    """
    Load a PDF, chunk it, embed it, and persist via VectorFactory.

    The vector provider (Chroma / FAISS / Pinecone) is determined
    entirely by general_settings["vector_provider"] — this function
    does not need to know which one is active.

    Parameters
    ----------
    pdf_path   : path to the PDF file
    embeddings : pre-loaded embedding model from app.state

    Returns
    -------
    (vector_store, chunk_count)
    """
    path = Path(pdf_path)
    if not path.exists():
        logger.error(f"INGEST | File not found: {pdf_path}")
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    size_kb = path.stat().st_size // 1024
    logger.info(f"INGEST | Loading: {path.name} ({size_kb} KB)")

    loader   = PyPDFLoader(str(path))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    try:
        chunks = loader.load_and_split(text_splitter=splitter)
    except Exception as e:
        logger.error(f"INGEST | PDF loading/splitting failed: {e}")
        raise

    logger.info(f"INGEST | {len(chunks)} chunks created from {path.name}")

    try:
        vector_store = VectorFactory.create(chunks, embeddings)
    except Exception as e:
        logger.error(f"INGEST | VectorFactory.create failed: {e}")
        raise

    logger.info(f"INGEST | Complete — {len(chunks)} chunks stored")
    return vector_store, len(chunks)


def load_existing(
    embeddings: HuggingFaceEmbeddings,
) -> VectorStore | None:
    """
    Load an existing vector store via VectorFactory.
    Returns None if no store has been built yet.
    """
    try:
        store = VectorFactory.load(embeddings)
        if store:
            logger.info("INGEST | Existing vector store loaded")
        else:
            logger.warning("INGEST | No existing vector store found")
        return store
    except Exception as e:
        logger.error(f"INGEST | load_existing failed: {e}")
        raise
