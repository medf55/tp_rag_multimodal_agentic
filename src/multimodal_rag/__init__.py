"""Package RAG multimodal et agentique (CLIP + ChromaDB + LangGraph ReAct)."""

from multimodal_rag.agent import AgenticMultimodalRAG
from multimodal_rag.chunking import CLIPTextChunker
from multimodal_rag.config import Settings, get_settings
from multimodal_rag.embeddings import CLIPEmbedder
from multimodal_rag.extraction import compute_document_id, extract_pdf_content
from multimodal_rag.llm import MultimodalRAGPipeline, create_chat_model
from multimodal_rag.models import (
    AgentResponse,
    ExtractionResult,
    IndexingReport,
    RAGResponse,
    RetrievedSource,
    ToolCallTrace,
)
from multimodal_rag.retriever import MultimodalRetriever
from multimodal_rag.tools import create_rag_tools
from multimodal_rag.vector_store import MultimodalVectorStore

__all__ = [
    "AgentResponse",
    "AgenticMultimodalRAG",
    "CLIPEmbedder",
    "CLIPTextChunker",
    "ExtractionResult",
    "IndexingReport",
    "MultimodalRAGPipeline",
    "MultimodalRetriever",
    "MultimodalVectorStore",
    "RAGResponse",
    "RetrievedSource",
    "Settings",
    "ToolCallTrace",
    "compute_document_id",
    "create_chat_model",
    "create_rag_tools",
    "extract_pdf_content",
    "get_settings",
]
