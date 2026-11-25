"""Tests End-to-End de l'agent LangGraph ReAct (Étapes 5, 6 et 7 du TP)."""

from __future__ import annotations

from pathlib import Path

import pytest

from multimodal_rag.agent import AgenticMultimodalRAG
from multimodal_rag.chunking import CLIPTextChunker
from multimodal_rag.config import PROJECT_ROOT, Settings
from multimodal_rag.embeddings import CLIPEmbedder
from multimodal_rag.extraction import extract_pdf_content
from multimodal_rag.llm import MultimodalRAGPipeline, create_chat_model
from multimodal_rag.retriever import MultimodalRetriever
from multimodal_rag.vector_store import MultimodalVectorStore

PDF_PATH = PROJECT_ROOT / "attention is all you need.pdf"


@pytest.fixture(scope="module")
def agent_env(tmp_path_factory: pytest.TempPathFactory):
    tmp_dir = tmp_path_factory.mktemp("agent_e2e")
    settings = Settings(
        pdf_path=PDF_PATH,
        output_dir=tmp_dir / "resultats_pdf",
        chroma_dir=tmp_dir / "chroma_pdf",
    )
    extraction = extract_pdf_content(settings.pdf_path, settings.output_dir)
    embedder = CLIPEmbedder(
        model_name=settings.clip_model_name,
        pretrained=settings.clip_pretrained,
    )
    store = MultimodalVectorStore(
        chroma_dir=settings.chroma_dir,
        collection_name="test_agent_e2e",
        embedder=embedder,
    )
    store.index_extraction(
        extraction,
        CLIPTextChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap),
        batch_size=settings.batch_size,
    )
    retriever = MultimodalRetriever(store)
    llm = create_chat_model(settings, temperature=0.0)
    rag_pipeline = MultimodalRAGPipeline(
        retriever=retriever,
        llm=llm,
        max_images=settings.max_images_per_generation,
    )
    agent = AgenticMultimodalRAG(
        settings=settings,
        retriever=retriever,
        rag_pipeline=rag_pipeline,
        llm=llm,
    )
    return agent, settings.output_dir / "images"


def test_1_greeting_without_tool(agent_env) -> None:
    agent, _ = agent_env
    mermaid = agent.get_mermaid_diagram()
    assert "mind" in mermaid and "body" in mermaid

    resp = agent.ask("Bonjour ! Que peux-tu faire ?")
    assert len(resp.tool_calls) == 0
    assert len(resp.answer) > 20


def test_2_textual_question_with_citations(agent_env) -> None:
    agent, _ = agent_env
    resp = agent.ask(
        question=(
            "D'après Attention Is All You Need, pourquoi le Transformer "
            "utilise-t-il l'attention multi-têtes ?"
        ),
        type_resultat="texte",
        k=8,
    )
    tool_names = [tc.name for tc in resp.tool_calls]
    assert "retriever_tool" in tool_names
    assert len(resp.sources) > 0
    lower = resp.answer.lower()
    assert "sous-espace" in lower or "subspace" in lower or "représentation" in lower


def test_3_hybrid_figure_2_retrieval_and_analysis(agent_env) -> None:
    agent, _ = agent_env
    resp = agent.ask(
        question=(
            "Retrouve la figure 2 (Scaled Dot-Product Attention et Multi-Head Attention) "
            "dans Attention Is All You Need et explique les opérations représentées."
        ),
        type_resultat="image",
        k=3,
    )
    tool_names = [tc.name for tc in resp.tool_calls]
    assert "retriever_tool" in tool_names
    assert any(img["page"] == 4 for img in resp.images)
    lower = resp.answer.lower()
    assert "matmul" in lower or "softmax" in lower or "scale" in lower or "attention" in lower


def test_4_image_input_query(agent_env) -> None:
    agent, image_dir = agent_env
    page3_images = sorted(image_dir.glob("page_003_*.png"))
    assert page3_images

    resp = agent.ask(
        question=(
            "Explique les composants de cette architecture en t'appuyant sur "
            "les passages de Attention Is All You Need."
        ),
        image_path=page3_images[0],
        type_resultat="hybride",
        k=6,
    )
    tool_names = [tc.name for tc in resp.tool_calls]
    assert "retriever_tool" in tool_names
    lower = resp.answer.lower()
    assert "encodeur" in lower or "encoder" in lower
    assert "décodeur" in lower or "decoder" in lower


def test_5_absent_information_abstention(agent_env) -> None:
    agent, _ = agent_env
    resp = agent.ask(
        question=(
            "Selon Attention Is All You Need, quel score obtient GPT-4 "
            "sur le benchmark MMLU ?"
        ),
        type_resultat="texte",
        k=6,
    )
    tool_names = [tc.name for tc in resp.tool_calls]
    assert "retriever_tool" in tool_names
    lower = resp.answer.lower()
    assert any(
        marker in lower
        for marker in ["ne contient pas", "pas mentionné", "absent", "ne figure pas", "aucune information", "ne mentionne pas", "ne permet pas", "pas présent"]
    )
