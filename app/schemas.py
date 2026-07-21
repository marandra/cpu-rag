"""Pydantic models for request/response."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    procedure: str = Field(..., min_length=1, max_length=50)


class HealthResponse(BaseModel):
    status: str
    model: str
    profile: str
    procedures: list[str]


class ChunkEvent(BaseModel):
    text: str


class DoneEvent(BaseModel):
    usage: dict


class ErrorEvent(BaseModel):
    code: str
    detail: str
