import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import general_settings

CHUNK_SIZE    = general_settings['ingestion']['chunk_size']
CHUNK_OVERLAP = general_settings['ingestion']['chunk_overlap']
SEPARATORS    = general_settings['ingestion']['separators']


def ingest(pdf_path: str,
           persist_dir: str,
           embeddings: HuggingFaceEmbeddings) -> tuple[Chroma, int]:
    """
    Load a PDF, chunk it, embed it, save to ChromaDB.

    Parameters
    ----------
    pdf_path    : path to the PDF on disk
    persist_dir : where to save / update the ChromaDB store
    embeddings  : already-loaded embedding model — passed in so the
                  90MB model is not reloaded on every call

    Returns
    -------
    (vector_db, chunk_count)

    Why chunk_size=1000 and overlap=200?
        1000 tokens fits one full concept explanation.
        200 token overlap means a concept that spans a page boundary
        is not cut in half — its tail appears at the head of the
        next chunk so retrieval never misses it.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"[ingest] Loading {path.name}")
    loader   = PyPDFLoader(str(path))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = loader.load_and_split(text_splitter=splitter)
    print(f"[ingest] {len(chunks)} chunks created")

    print(f"[ingest] Embedding and storing in {persist_dir}")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print("[ingest] Complete")
    return vector_db, len(chunks)


def load_existing(persist_dir: str,
                  embeddings: HuggingFaceEmbeddings) -> Chroma | None:
    """
    Load a ChromaDB store that was already built.
    Returns None if it has not been built yet.
    """
    if not Path(persist_dir).exists():
        return None
    print(f"[ingest] Loading existing vector store from {persist_dir}")
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )