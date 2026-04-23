"""
Centralized Pydantic models for type safety across the system.
"""
from pydantic import BaseModel, Field
from typing import Optional


class QuestionRequest(BaseModel):
    """Request body for /ask endpoint."""
    question: str = Field(..., description="The question to ask the bot")
    session_id: str = Field("default", description="Session ID for conversation history")


class AnswerResponse(BaseModel):
    """Response body for /ask endpoint."""
    answer: str = Field(..., description="The bot's answer")
    session_id: str = Field(..., description="Session ID for tracking")


class IngestResponse(BaseModel):
    """Response body for /upload endpoint."""
    message: str = Field(..., description="Status message")
    filename: str = Field(..., description="Name of ingested file")
    chunks: int = Field(..., description="Number of chunks created")


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    database: str = Field(..., description="Database connection status")
    vector_store: str = Field(..., description="Vector store status")
    groq: str = Field(..., description="Groq API connection status")


class HomeResponse(BaseModel):
    """Response body for / endpoint."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    docs: str = Field(..., description="Documentation URL")


class BooksResponse(BaseModel):
    """Response body for /books endpoint."""
    books: list[str] = Field(..., description="List of ingested book titles")


class Message(BaseModel):
    """A single message in conversation history."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ToolCall(BaseModel):
    """Tool invocation metadata."""
    name: str = Field(..., description="Tool name")
    enabled: bool = Field(..., description="Whether tool is enabled")
    triggers: list[str] = Field(default_factory=list, description="Intent trigger phrases")
