"""Modèles de données Pydantic pour toutes les étapes du RAG multimodal."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    """Page textuelle extraite d'un document PDF."""

    page_number: int
    text: str
    text_path: Path


class ExtractedImage(BaseModel):
    """Image extraite et sauvegardée depuis une page PDF."""

    page_number: int
    xref: int
    path: Path
    width: int
    height: int
    ext: str


class ExtractionResult(BaseModel):
    """Résultat complet de l'extraction multimodale d'un PDF."""

    pdf_path: Path
    document_id: str
    total_pages: int
    pages: list[ExtractedPage]
    images: list[ExtractedImage]
    full_text_path: Path
    pages_without_text: list[int] = Field(default_factory=list)


class IndexingReport(BaseModel):
    """Compte-rendu d'indexation dans la base vectorielle ChromaDB."""

    document_id: str
    pdf_name: str
    text_chunks_indexed: int
    images_indexed: int
    total_collection_items: int


class RetrievedSource(BaseModel):
    """Source textuelle ou visuelle retrouvée par similarité cosinus."""

    reference: str
    id: str
    type: Literal["texte", "image"]
    pdf: str
    page: int
    source: str
    content: str
    similarity: float


class ToolCallTrace(BaseModel):
    """Trace d'un appel d'outil effectué par l'agent ReAct."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    output_preview: str = ""


class RAGResponse(BaseModel):
    """Réponse générée par le pipeline RAG classique."""

    question: str
    answer: str
    sources: list[RetrievedSource] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Réponse finale et trace complète de l'agent LangGraph ReAct."""

    question: str
    answer: str
    sources: list[RetrievedSource] = Field(default_factory=list)
    images: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    steps_count: int = 0
