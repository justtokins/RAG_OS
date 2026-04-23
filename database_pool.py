"""
Database connection pooling for efficient PostgreSQL management.
"""
import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

# Create a thread-safe connection pool
_connection_pool = None


def get_connection_pool():
    """Get or create the global connection pool."""
    global _connection_pool
    if _connection_pool is None:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise RuntimeError("DATABASE_URL not set in environment")
        
        _connection_pool = psycopg2.pool.SimpleConnectionPool(
            1,  # minconn
            10,  # maxconn
            db_url,
        )
    return _connection_pool


def get_connection():
    """Get a connection from the pool."""
    pool = get_connection_pool()
    return pool.getconn()


def close_connection(conn):
    """Return a connection to the pool."""
    pool = get_connection_pool()
    pool.putconn(conn)


def close_all_connections():
    """Close all connections in the pool. Call on shutdown."""
    global _connection_pool
    if _connection_pool:
        _connection_pool.closeall()
        _connection_pool = None


def setup_db():
    """Create all tables if they do not exist. Called once at startup."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id         SERIAL PRIMARY KEY,
                session_id TEXT      NOT NULL,
                role       TEXT      NOT NULL,
                content    TEXT      NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS ingested_books (
                id          SERIAL PRIMARY KEY,
                filename    TEXT      NOT NULL,
                chunk_count INTEGER   NOT NULL,
                ingested_at TIMESTAMP DEFAULT NOW()
            );
        """)
        conn.commit()
        print("[db] Tables ready")
    finally:
        close_connection(conn)


def save_message(session_id: str, role: str, content: str):
    """Save a message to chat history."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_history (session_id, role, content) "
            "VALUES (%s, %s, %s)",
            (session_id, role, content),
        )
        conn.commit()
    finally:
        close_connection(conn)


def get_history(session_id: str, limit: int = 6) -> list:
    """Fetch chat history for a session."""
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
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    finally:
        close_connection(conn)


def log_ingestion(filename: str, chunk_count: int):
    """Log a PDF ingestion event."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ingested_books (filename, chunk_count) "
            "VALUES (%s, %s)",
            (filename, chunk_count),
        )
        conn.commit()
    finally:
        close_connection(conn)


def list_books() -> list:
    """Fetch all ingested books."""
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
                "filename": r[0],
                "chunks": r[1],
                "ingested_at": r[2].isoformat() if r[2] else None,
            }
            for r in rows
        ]
    finally:
        close_connection(conn)
