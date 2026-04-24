"""
Database connection pooling + encrypted persistence.

Encryption contract:
    WRITE — content is encrypted before INSERT. What lives in
            PostgreSQL is ciphertext, never plaintext.
    READ  — content is decrypted after SELECT. The rest of the
            system only ever sees plaintext.

This means:
    - A compromised database dump reveals nothing readable.
    - A compromised DB connection (MITM at the wire layer) sees ciphertext.
    - Only the application holding ENCRYPTION_KEY can read the data.

Fallback behaviour:
    If decryption fails on a row (e.g. old unencrypted rows from before
    this module was added), the raw value is returned with a WARNING log.
    This prevents the entire history fetch from crashing on legacy data.
"""
import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

from encryption import get_encryption
from logger import get_logger

load_dotenv()

logger = get_logger()

# ── Connection pool ───────────────────────────────────────────────────────────
_connection_pool = None


def get_connection_pool():
    """Lazily create the global connection pool."""
    global _connection_pool
    if _connection_pool is None:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise RuntimeError("DATABASE_URL not set in environment")
        _connection_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=db_url,
        )
        logger.info("DB | Connection pool created (min=1, max=10)")
    return _connection_pool


def get_connection():
    return get_connection_pool().getconn()


def close_connection(conn):
    get_connection_pool().putconn(conn)


def close_all_connections():
    """Return all connections to pool and destroy it. Call on shutdown."""
    global _connection_pool
    if _connection_pool:
        _connection_pool.closeall()
        _connection_pool = None
        logger.info("DB | Connection pool closed")


# ── Schema setup ──────────────────────────────────────────────────────────────

def setup_db():
    """
    Create tables if they do not exist. Called once at startup.

    chat_history.content stores ENCRYPTED ciphertext — never plaintext.
    ingested_books.filename stores the original filename in plaintext
    (not sensitive — it is just a book title for the /books endpoint).
    """
    conn = get_connection()
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
        logger.info("DB | Tables verified / created")
    except Exception as e:
        logger.error(f"DB | setup_db failed: {e}")
        raise
    finally:
        close_connection(conn)


# ── Chat history ──────────────────────────────────────────────────────────────

def save_message(session_id: str, role: str, content: str):
    """
    Encrypt content and persist to PostgreSQL.

    The plaintext content is encrypted to ciphertext before the
    INSERT. If you inspect the database directly you will see
    Fernet tokens, not readable text.
    """
    enc = get_encryption()
    try:
        encrypted_content = enc.encrypt(content)
    except Exception as e:
        logger.error(f"DB | Encryption failed for session={session_id}: {e}")
        raise

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_history (session_id, role, content) "
            "VALUES (%s, %s, %s)",
            (session_id, role, encrypted_content),
        )
        conn.commit()
        logger.debug(f"DB | Message saved | session={session_id} | role={role}")
    except Exception as e:
        logger.error(f"DB | save_message failed: {e}")
        raise
    finally:
        close_connection(conn)


def get_history(session_id: str, limit: int = 6) -> list:
    """
    Fetch and decrypt the most recent `limit` messages for a session.

    Decryption failure on a row logs a WARNING and returns the raw
    ciphertext as a fallback — this handles legacy unencrypted rows
    without crashing the entire history fetch.
    """
    enc = get_connection_pool()   # just checking pool is alive
    enc = get_encryption()

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content FROM chat_history "
            "WHERE session_id = %s "
            "ORDER BY created_at DESC LIMIT %s",
            (session_id, limit),
        )
        rows = cur.fetchall()
    except Exception as e:
        logger.error(f"DB | get_history failed: {e}")
        raise
    finally:
        close_connection(conn)

    history = []
    for role, ciphertext in reversed(rows):
        try:
            plaintext = enc.decrypt(ciphertext)
        except Exception:
            logger.warning(
                f"DB | Decryption failed for session={session_id} role={role} "
                f"— returning raw value (possible legacy unencrypted row)"
            )
            plaintext = ciphertext   # graceful fallback

        history.append({"role": role, "content": plaintext})

    logger.debug(f"DB | History fetched | session={session_id} | rows={len(history)}")
    return history


# ── Book ingestion log ────────────────────────────────────────────────────────

def log_ingestion(filename: str, chunk_count: int):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ingested_books (filename, chunk_count) VALUES (%s, %s)",
            (filename, chunk_count),
        )
        conn.commit()
        logger.info(f"DB | Ingestion logged | file={filename} | chunks={chunk_count}")
    except Exception as e:
        logger.error(f"DB | log_ingestion failed: {e}")
        raise
    finally:
        close_connection(conn)


def list_books() -> list:
    conn = get_connection()
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
        logger.error(f"DB | list_books failed: {e}")
        raise
    finally:
        close_connection(conn)
