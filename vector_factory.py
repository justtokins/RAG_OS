"""
vector_factory.py — Abstract vector store creation and loading.

Why a factory?
    Without this, every file that needs a vector store imports Chroma
    directly and knows how to initialise it. Swapping to FAISS or
    Pinecone means touching every one of those files.

    With a factory, the rest of the system calls:
        VectorFactory.create(embeddings, chunks)
        VectorFactory.load(embeddings)

    Swapping the provider = change one value in general_settings.json:
        "vector_provider": "faiss"   ← was "chroma"

    Nothing else changes.

Supported providers (add more by extending _REGISTRY):
    chroma  — local persistent vector store (default, no extra deps)
    faiss   — in-memory / on-disk via Facebook AI Similarity Search
    pinecone — cloud-hosted (requires API key and extra config)

Adding a new provider:
    1. Write a module under vector_providers/<name>.py
       implementing create() and load() with the same signatures.
    2. Register it in _REGISTRY below.
    3. Add any required config keys to general_settings.json.
    That is all. No other file needs to change.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from config_loader import general_settings
from logger import get_logger

logger = get_logger()


# ── Provider protocol ─────────────────────────────────────────────────────────
@runtime_checkable
class VectorProvider(Protocol):
    """
    Interface every provider module must satisfy.
    Python's Protocol means we get duck-typing with static analysis.
    """

    def create(
        self,
        documents: list[Document],
        embeddings: HuggingFaceEmbeddings,
        **kwargs,
    ) -> VectorStore:
        """Embed documents and persist a new vector store."""
        ...

    def load(
        self,
        embeddings: HuggingFaceEmbeddings,
        **kwargs,
    ) -> VectorStore | None:
        """Load an existing vector store. Returns None if not found."""
        ...


# ── Provider implementations ──────────────────────────────────────────────────
class _ChromaProvider:
    """
    Local persistent vector store using ChromaDB.
    No API key required. Stores data in a local directory.
    Best for: development, on-premise deployment, privacy-first setups.
    """

    def create(
        self,
        documents: list[Document],
        embeddings: HuggingFaceEmbeddings,
        persist_dir: str = "./vector_store",
        **kwargs,
    ) -> VectorStore:
        from langchain_community.vectorstores import Chroma
        logger.info(f"VECTOR_FACTORY | Chroma | Creating from {len(documents)} docs → {persist_dir}")
        store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
        logger.info("VECTOR_FACTORY | Chroma | Store created and persisted")
        return store

    def load(
        self,
        embeddings: HuggingFaceEmbeddings,
        persist_dir: str = "./vector_store",
        **kwargs,
    ) -> VectorStore | None:
        from langchain_community.vectorstores import Chroma
        from pathlib import Path
        if not Path(persist_dir).exists():
            logger.warning(f"VECTOR_FACTORY | Chroma | No store at {persist_dir}")
            return None
        logger.info(f"VECTOR_FACTORY | Chroma | Loading from {persist_dir}")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )


class _FAISSProvider:
    """
    FAISS — Facebook AI Similarity Search.
    In-memory during session, saved to disk as an index file.
    Best for: fast similarity search, large datasets, CPU-only servers.
    Requires: pip install faiss-cpu  (or faiss-gpu for GPU servers)
    """

    def create(
        self,
        documents: list[Document],
        embeddings: HuggingFaceEmbeddings,
        index_path: str = "./faiss_index",
        **kwargs,
    ) -> VectorStore:
        from langchain_community.vectorstores import FAISS
        logger.info(f"VECTOR_FACTORY | FAISS | Creating index from {len(documents)} docs")
        store = FAISS.from_documents(documents, embeddings)
        store.save_local(index_path)
        logger.info(f"VECTOR_FACTORY | FAISS | Index saved to {index_path}")
        return store

    def load(
        self,
        embeddings: HuggingFaceEmbeddings,
        index_path: str = "./faiss_index",
        **kwargs,
    ) -> VectorStore | None:
        from langchain_community.vectorstores import FAISS
        from pathlib import Path
        if not Path(index_path).exists():
            logger.warning(f"VECTOR_FACTORY | FAISS | No index at {index_path}")
            return None
        logger.info(f"VECTOR_FACTORY | FAISS | Loading index from {index_path}")
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )


class _PineconeProvider:
    """
    Pinecone — managed cloud vector database.
    Best for: production deployments, very large corpora, multi-tenant SaaS.
    Requires: pip install pinecone-client
              PINECONE_API_KEY and PINECONE_INDEX_NAME in .env
    """

    def create(
        self,
        documents: list[Document],
        embeddings: HuggingFaceEmbeddings,
        **kwargs,
    ) -> VectorStore:
        import os
        from langchain_community.vectorstores import Pinecone
        import pinecone

        api_key    = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not api_key or not index_name:
            raise EnvironmentError(
                "PINECONE_API_KEY and PINECONE_INDEX_NAME must be set for Pinecone provider"
            )
        pinecone.init(api_key=api_key)
        logger.info(f"VECTOR_FACTORY | Pinecone | Upserting {len(documents)} docs → {index_name}")
        store = Pinecone.from_documents(documents, embeddings, index_name=index_name)
        logger.info("VECTOR_FACTORY | Pinecone | Upsert complete")
        return store

    def load(
        self,
        embeddings: HuggingFaceEmbeddings,
        **kwargs,
    ) -> VectorStore | None:
        import os
        from langchain_community.vectorstores import Pinecone
        import pinecone

        api_key    = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not api_key or not index_name:
            logger.warning("VECTOR_FACTORY | Pinecone | Missing env vars — cannot load")
            return None
        pinecone.init(api_key=api_key)
        logger.info(f"VECTOR_FACTORY | Pinecone | Connecting to index {index_name}")
        return Pinecone.from_existing_index(index_name, embeddings)


# ── Provider registry ─────────────────────────────────────────────────────────
# Add new providers here. Key must match general_settings["vector_provider"].
_REGISTRY: dict[str, VectorProvider] = {
    "chroma":   _ChromaProvider(),
    "faiss":    _FAISSProvider(),
    "pinecone": _PineconeProvider(),
}


# ── Public factory interface ──────────────────────────────────────────────────
class VectorFactory:
    """
    Public interface. The rest of the system only uses this class —
    never the provider implementations directly.

    Usage:
        store = VectorFactory.create(embeddings, chunks)
        store = VectorFactory.load(embeddings)

    To swap providers: change general_settings.json, restart server.
    """

    @staticmethod
    def _get_provider() -> VectorProvider:
        provider_name = general_settings.get("vector_provider", "chroma")
        provider = _REGISTRY.get(provider_name)
        if provider is None:
            raise ValueError(
                f"Unknown vector provider '{provider_name}'. "
                f"Available: {list(_REGISTRY.keys())}"
            )
        logger.debug(f"VECTOR_FACTORY | Using provider: {provider_name}")
        return provider

    @staticmethod
    def _get_kwargs() -> dict:
        """Pull provider-specific kwargs from general_settings."""
        return {
            "persist_dir": general_settings["embedding"].get("vector_db_dir", "./vector_store"),
            "index_path":  general_settings["embedding"].get("vector_db_dir", "./faiss_index"),
        }

    @classmethod
    def create(
        cls,
        documents: list[Document],
        embeddings: HuggingFaceEmbeddings,
    ) -> VectorStore:
        """
        Embed documents and persist a new vector store.
        Provider is determined by general_settings["vector_provider"].
        """
        provider = cls._get_provider()
        return provider.create(documents, embeddings, **cls._get_kwargs())

    @classmethod
    def load(
        cls,
        embeddings: HuggingFaceEmbeddings,
    ) -> VectorStore | None:
        """
        Load an existing vector store from disk / cloud.
        Returns None if no store exists yet.
        """
        provider = cls._get_provider()
        return provider.load(embeddings, **cls._get_kwargs())
