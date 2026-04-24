"""
Centralised Pydantic models.

Every request body, response body, and internal data structure
is defined here. Pydantic enforces types at runtime — if a field
receives the wrong type, a 422 Validation Error is raised before
any business logic runs.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ── Request models ────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """POST /ask request body."""
    question:   str = Field(...,         min_length=1, description="Question to ask")
    session_id: str = Field("default",               description="Conversation session ID")
    format:     str = Field("markdown",              description="Output format: markdown | plain | html")

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()

    @field_validator("format")
    @classmethod
    def valid_format(cls, v: str) -> str:
        allowed = {"markdown", "plain", "html"}
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}")
        return v


# ── Response models ───────────────────────────────────────────────────────────

class AnswerResponse(BaseModel):
    """POST /ask response body."""
    answer:     str = Field(..., description="Formatted answer from the assistant")
    session_id: str = Field(..., description="Session ID echoed back")
    intent:     str = Field("unknown", description="Intent detected by agent")
    tool_used:  str = Field("unknown", description="Tool selected by agent")


class IngestResponse(BaseModel):
    """POST /upload response body."""
    message:    str = Field(..., description="Status message")
    filename:   str = Field(..., description="Name of ingested file")
    chunks:     int = Field(..., description="Number of chunks created")


class HealthResponse(BaseModel):
    """GET /health response body."""
    database:     str = Field(..., description="Database connection status")
    vector_store: str = Field(..., description="Vector store status")
    groq:         str = Field(..., description="Groq API connection status")


class HomeResponse(BaseModel):
    """GET / response body."""
    status:  str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    docs:    str = Field(..., description="Swagger docs URL")


class BookRecord(BaseModel):
    """A single ingested book record."""
    filename:    str            = Field(..., description="Original filename")
    chunks:      int            = Field(..., description="Chunks in vector store")
    ingested_at: Optional[str]  = Field(None, description="ISO timestamp")


class BooksResponse(BaseModel):
    """GET /books response body."""
    books: list[BookRecord] = Field(..., description="All ingested books")


# ── Internal data structures ──────────────────────────────────────────────────

class Message(BaseModel):
    """One turn of conversation history."""
    role:    Literal["user", "assistant"] = Field(..., description="Speaker")
    content: str                          = Field(..., description="Message text")


class ToolCall(BaseModel):
    """Metadata about a tool invocation — used for logging and tracing."""
    name:     str       = Field(..., description="Tool function name")
    enabled:  bool      = Field(..., description="Whether the tool is active")
    triggers: list[str] = Field(default_factory=list, description="Intent triggers")


class AgentDecision(BaseModel):
    """
    Captures everything the agent decided for one request.
    Returned from route() so query.py can log it and include
    tool_used and intent in the API response.
    """
    intent:   str = Field(..., description="Detected intent")
    tool:     str = Field(..., description="Tool selected")
    context:  str = Field(..., description="Retrieved context string")
