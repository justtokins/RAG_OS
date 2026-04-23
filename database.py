import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def setup_db():
    """Create all tables if they do not exist. Called once at startup."""
    conn = get_connection()
    cur  = conn.cursor()
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
    conn.close()
    print("[db] Tables ready")


def save_message(session_id: str, role: str, content: str):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO chat_history (session_id, role, content) "
        "VALUES (%s, %s, %s)",
        (session_id, role, content),
    )
    conn.commit()
    conn.close()


def get_history(session_id: str, limit: int = 6) -> list:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        "SELECT role, content FROM chat_history "
        "WHERE session_id = %s "
        "ORDER BY created_at DESC LIMIT %s",
        (session_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


def log_ingestion(filename: str, chunk_count: int):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO ingested_books (filename, chunk_count) "
        "VALUES (%s, %s)",
        (filename, chunk_count),
    )
    conn.commit()
    conn.close()


def list_books() -> list:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        "SELECT filename, chunk_count, ingested_at "
        "FROM ingested_books ORDER BY ingested_at DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {"filename": r[0], "chunks": r[1], "ingested_at": str(r[2])}
        for r in rows
    ]