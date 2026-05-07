"""
database_pool.py — PostgreSQL with async write support.

Two classes of operations:

WRITES (async) — save_message, log_ingestion
    Called after the LLM responds. The user already has their answer —
    there is no reason to make them wait while we write to the DB.
    asyncio.to_thread() runs the psycopg2 call in a thread pool,
    releasing the event loop immediately.

    Fire-and-forget pattern for save_message:
        asyncio.create_task(async_save_message(...))
    The task runs in the background. If it fails, it logs an error
    but does not affect the response the user already received.

READS (sync) — get_history, list_books
    Called before the LLM responds — the prompt depends on history.
    Must complete before we can proceed. Kept synchronous and run
    via asyncio.to_thread() at the call site in query.py.

Why asyncio.to_thread() instead of asyncpg?
    asyncpg requires rewriting all SQL to use its parameter style
    ($1, $2 vs %s) and a separate connection pool API. asyncio.to_thread()
    gives us async behaviour with zero migration cost — the same
    psycopg2 pool, same SQL, same error handling. The thread pool used
    by asyncio.to_thread() defaults to ThreadPoolExecutor(max_workers=32)
    which is more than enough for concurrent DB writes.

Encryption:
    All chat content is encrypted before INSERT and decrypted after SELECT.
    DB admins see ciphertext. The ENCRYPTION_KEY never touches the database.
"""
import asyncio
import os

import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv


from .encryption import get_encryption
from .logger import get_logger

load_dotenv()
logger = get_logger()

_pool: pool.SimpleConnectionPool | None = None


# ── Connection pool ───────────────────────────────────────────────────────────

def get_pool() -> pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise RuntimeError("DATABASE_URL not set")
        
        _pool = pool.SimpleConnectionPool(minconn=1, maxconn=10, dsn=db_url)
        logger.info("DB | Connection pool created (min=1 max=10)")

    # Explicitly tell Pylance: "If we got here, _pool is definitely the pool"
    if _pool is None:
        raise RuntimeError("Failed to initialize database pool")
        
    return _pool


def _conn():
    return get_pool().getconn()


def _release(conn):
    get_pool().putconn(conn)


def close_all_connections():
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None
        logger.info("DB | Pool closed")


# ── Schema ────────────────────────────────────────────────────────────────────

def setup_db():
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id         SERIAL    PRIMARY KEY,
                session_id TEXT      NOT NULL,
                role       TEXT      NOT NULL,
                content    TEXT      NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS ingested_books (
                id          SERIAL    PRIMARY KEY,
                filename    TEXT      NOT NULL,
                chunk_count INTEGER   NOT NULL,
                ingested_at TIMESTAMP DEFAULT NOW()
            );
        """)
        conn.commit()
        logger.info("DB | Tables verified")
    except Exception as e:
        logger.error(f"DB | setup_db failed: {e}")
        raise
    finally:
        _release(conn)


# ── Sync write primitives (called by async wrappers) ─────────────────────────

def _sync_save_message(session_id: str, role: str, content: str):
    """Encrypt and insert one message. Runs in thread pool."""
    enc              = get_encryption()
    encrypted        = enc.encrypt(content)

    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_history (session_id, role, content) "
            "VALUES (%s, %s, %s)",
            (session_id, role, encrypted),
        )
        conn.commit()
        logger.debug(f"DB | Saved | session={session_id} role={role}")
    except Exception as e:
        logger.error(f"DB | _sync_save_message failed: {e}")
        raise
    finally:
        _release(conn)


def _sync_log_ingestion(filename: str, chunk_count: int):
    """Log one ingestion event. Runs in thread pool."""
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ingested_books (filename, chunk_count) VALUES (%s, %s)",
            (filename, chunk_count),
        )
        conn.commit()
        logger.info(f"DB | Ingestion logged | {filename} | {chunk_count} chunks")
    except Exception as e:
        logger.error(f"DB | _sync_log_ingestion failed: {e}")
        raise
    finally:
        _release(conn)


# ── Async write wrappers ──────────────────────────────────────────────────────

async def async_save_message(session_id: str, role: str, content: str):
    """
    Non-blocking message save.

    asyncio.to_thread() submits _sync_save_message to the default
    ThreadPoolExecutor and returns control to the event loop immediately.
    The DB write completes in the background.

    Usage (fire-and-forget — most common):
        asyncio.create_task(async_save_message(session_id, role, content))

    Usage (wait for confirmation — use only if you need the write to
    complete before the next operation):
        await async_save_message(session_id, role, content)
    """
    await asyncio.to_thread(_sync_save_message, session_id, role, content)


async def async_log_ingestion(filename: str, chunk_count: int):
    """Non-blocking ingestion log."""
    await asyncio.to_thread(_sync_log_ingestion, filename, chunk_count)


# ── Sync read operations (awaited at call site via to_thread) ─────────────────

def _sync_get_history(session_id: str, limit: int) -> list[dict]:
    """
    Fetch and decrypt conversation history. Sync — runs in thread pool
    via asyncio.to_thread() in query.py.
    """
    enc  = get_encryption()
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content FROM chat_history "
            "WHERE session_id = %s ORDER BY created_at DESC LIMIT %s",
            (session_id, limit),
        )
        rows = cur.fetchall()
    except Exception as e:
        logger.error(f"DB | _sync_get_history failed: {e}")
        raise
    finally:
        _release(conn)

    history = []
    for role, ciphertext in reversed(rows):
        try:
            plaintext = enc.decrypt(ciphertext)
        except Exception:
            logger.warning(
                f"DB | Decryption failed session={session_id} role={role} "
                f"— returning raw (possible legacy unencrypted row)"
            )
            plaintext = ciphertext
        history.append({"role": role, "content": plaintext})

    logger.debug(f"DB | History | session={session_id} | rows={len(history)}")
    return history


async def get_history(session_id: str, limit: int = 6) -> list[dict]:
    """
    Async wrapper for history fetch.
    Awaited in query.py so the event loop is not blocked during the DB read.
    """
    return await asyncio.to_thread(_sync_get_history, session_id, limit)


def _sync_list_books() -> list[dict]:
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT filename, chunk_count, ingested_at "
            "FROM ingested_books ORDER BY ingested_at DESC"
        )
        rows = cur.fetchall()
        return [
            {
                "filename":    r[0],
                "chunks":      r[1],
                "ingested_at": r[2].isoformat() if r[2] else None,
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"DB | _sync_list_books failed: {e}")
        raise
    finally:
        _release(conn)


async def list_books() -> list[dict]:
    return await asyncio.to_thread(_sync_list_books)


# ── Backward-compatible sync aliases ─────────────────────────────────────────
# Some parts of the system (background tasks that cannot await) still use
# the sync versions directly. Keep these available.

def save_message(session_id: str, role: str, content: str):
    """Sync alias — use only in non-async contexts (background threads)."""
    _sync_save_message(session_id, role, content)


def log_ingestion(filename: str, chunk_count: int):
    """Sync alias — use only in non-async contexts (background threads)."""
    _sync_log_ingestion(filename, chunk_count)