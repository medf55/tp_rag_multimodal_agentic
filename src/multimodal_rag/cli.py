"""Interface en ligne de commande (CLI) pour le pipeline RAG multimodal agentique."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from multimodal_rag.agent import AgenticMultimodalRAG
from multimodal_rag.chunking import CLIPTextChunker
from multimodal_rag.config import Settings, get_settings
from multimodal_rag.embeddings import CLIPEmbedder
from multimodal_rag.extraction import extract_pdf_content
from multimodal_rag.llm import MultimodalRAGPipeline, create_chat_model
from multimodal_rag.retriever import MultimodalRetriever
from multimodal_rag.vector_store import MultimodalVectorStore


def _build_store(settings: Settings) -> MultimodalVectorStore:
    embedder = CLIPEmbedder(
        model_name=settings.clip_model_name,
        pretrained=settings.clip_pretrained,
    )
    return MultimodalVectorStore(
        chroma_dir=settings.chroma_dir,
        collection_name=settings.collection_name,
        embedder=embedder,
    )


def _cmd_index(settings: Settings, pdf_override: str | None) -> int:
    pdf_path = Path(pdf_override) if pdf_override else settings.pdf_path
    extraction = extract_pdf_content(pdf_path=pdf_path, output_dir=settings.output_dir)
    chunker = CLIPTextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    store = _build_store(settings)
    report = store.index_extraction(
        extraction=extraction,
        chunker=chunker,
        batch_size=settings.batch_size,
    )
    print(f"PDF             : {extraction.pdf_path}")
    print(f"Document ID     : {report.document_id}")
    print(f"Pages extraites : {extraction.total_pages}")
    print(f"Images extraites: {len(extraction.images)}")
    print(f"Chunks texte    : {report.text_chunks_indexed}")
    print(f"Images indexées : {report.images_indexed}")
    print(f"Total ChromaDB  : {report.total_collection_items}")
    return 0


def _cmd_search(
    settings: Settings,
    text: str | None,
    image: str | None,
    result_type: str,
    k: int,
) -> int:
    store = _build_store(settings)
    retriever = MultimodalRetriever(store)
    results = retriever.search(
        texte=text,
        image=image,
        type_resultat=result_type,  # type: ignore[arg-type]
        k=k,
    )
    for src in results:
        print(
            f"[{src.reference}] {src.type.upper()} | {src.pdf} | p.{src.page} "
            f"| sim={src.similarity:.3f}"
        )
        if src.type == "texte":
            print(f"    {src.content[:180]}")
        else:
            print(f"    image: {src.source}")
    return 0


def _cmd_ask(
    settings: Settings,
    text: str | None,
    image: str | None,
    result_type: str,
    k: int,
) -> int:
    store = _build_store(settings)
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
    response = agent.ask(
        question=text,
        image_path=image,
        type_resultat=result_type,  # type: ignore[arg-type]
        k=k,
    )
    print("=== APPELS D'OUTILS ===")
    if not response.tool_calls:
        print("(aucun outil appelé)")
    for tc in response.tool_calls:
        print(f"- {tc.name}({tc.args})")
    print("\n=== RÉPONSE AGENT ===")
    print(response.answer)
    if response.images:
        print("\n=== IMAGES RETROUVÉES ===")
        for img in response.images:
            print(f"- p.{img['page']} : {img['chemin']} (sim={img['similarity']:.3f})")
    return 0


def _cmd_graph(settings: Settings) -> int:
    # Affiche le graphe LangGraph avec un LLM factice si aucune clé API n'est requise pour le schéma.
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class _DummyStore(MultimodalVectorStore):
        def __init__(self) -> None:
            pass

    class _DummyRetriever(MultimodalRetriever):
        def __init__(self) -> None:
            pass

    class _DummyRAG(MultimodalRAGPipeline):
        def __init__(self) -> None:
            pass

    class _DummyChatModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):  # type: ignore[override]
            return self

    fake_llm = _DummyChatModel(messages=iter([AIMessage(content="ok")]))
    agent = AgenticMultimodalRAG(
        settings=settings,
        retriever=_DummyRetriever(),
        rag_pipeline=_DummyRAG(),
        llm=fake_llm,
    )
    print(agent.get_mermaid_diagram())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multimodal-rag",
        description="CLI du TP RAG Multimodal Agentique (CLIP + ChromaDB + LangGraph)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_index = subparsers.add_parser("index", help="Extraire et indexer un PDF dans ChromaDB")
    p_index.add_argument("--pdf", type=str, default=None, help="Chemin du PDF à indexer")

    p_search = subparsers.add_parser("search", help="Recherche vectorielle multimodale")
    p_search.add_argument("--text", type=str, default=None, help="Requête textuelle")
    p_search.add_argument("--image", type=str, default=None, help="Chemin d'image requête")
    p_search.add_argument(
        "--type",
        choices=["texte", "image", "hybride", "tous"],
        default="hybride",
        help="Filtrage par modalité",
    )
    p_search.add_argument("-k", type=int, default=5, help="Nombre de résultats")

    p_ask = subparsers.add_parser("ask", help="Interroger l'agent ReAct LangGraph")
    p_ask.add_argument("--text", type=str, default=None, help="Question utilisateur")
    p_ask.add_argument("--image", type=str, default=None, help="Chemin d'image en entrée")
    p_ask.add_argument(
        "--type",
        choices=["texte", "image", "hybride", "tous"],
        default="hybride",
        help="Mode de recherche suggéré à l'agent",
    )
    p_ask.add_argument("-k", type=int, default=6, help="Nombre de sources demandées")

    subparsers.add_parser("graph", help="Afficher le diagramme Mermaid du graphe LangGraph")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.command == "index":
        return _cmd_index(settings, args.pdf)
    if args.command == "search":
        return _cmd_search(settings, args.text, args.image, args.type, args.k)
    if args.command == "ask":
        return _cmd_ask(settings, args.text, args.image, args.type, args.k)
    if args.command == "graph":
        return _cmd_graph(settings)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
