"""Tests d'intégration pour CLIP (Étape 2), ChromaDB (Étape 3) et Retriever (Étape 4)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from multimodal_rag.chunking import CLIPTextChunker
from multimodal_rag.config import PROJECT_ROOT
from multimodal_rag.embeddings import CLIPEmbedder
from multimodal_rag.extraction import extract_pdf_content
from multimodal_rag.retriever import MultimodalRetriever
from multimodal_rag.vector_store import MultimodalVectorStore

PDF_PATH = PROJECT_ROOT / "attention is all you need.pdf"


@pytest.fixture(scope="module")
def clip_embedder() -> CLIPEmbedder:
    return CLIPEmbedder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")


def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def test_clip_embeddings_dimension_and_normalization(
    clip_embedder: CLIPEmbedder,
    tmp_path: Path,
) -> None:
    extraction = extract_pdf_content(PDF_PATH, tmp_path / "extract")
    sample_img = extraction.images[0].path

    text_vecs = clip_embedder.encode_texts(["Multi-Head Attention in the Transformer"])
    img_vec = clip_embedder.encode_image(sample_img)

    assert len(text_vecs) == 1
    assert len(text_vecs[0]) == 512
    assert len(img_vec) == 512
    assert _l2_norm(text_vecs[0]) == pytest.approx(1.0, abs=1e-4)
    assert _l2_norm(img_vec) == pytest.approx(1.0, abs=1e-4)


def test_chromadb_idempotent_indexing_and_multimodal_retrieval(
    clip_embedder: CLIPEmbedder,
    tmp_path: Path,
) -> None:
    extraction = extract_pdf_content(PDF_PATH, tmp_path / "extract")
    chunker = CLIPTextChunker(chunk_size=60, chunk_overlap=10)
    store = MultimodalVectorStore(
        chroma_dir=tmp_path / "chroma",
        collection_name="test_pdf_clip",
        embedder=clip_embedder,
    )

    report_1 = store.index_extraction(extraction, chunker, batch_size=64)
    assert report_1.text_chunks_indexed > 50
    assert report_1.images_indexed >= 3
    count_after_first = store.count()
    assert count_after_first == report_1.text_chunks_indexed + report_1.images_indexed

    # Idempotence : réindexer le même PDF conserve exactement le même nombre d'éléments.
    report_2 = store.index_extraction(extraction, chunker, batch_size=64)
    assert report_2.total_collection_items == count_after_first
    assert store.count() == count_after_first

    retriever = MultimodalRetriever(store)

    # 1. Recherche texte uniquement
    text_hits = retriever.search(
        texte="Why does the Transformer use multi-head attention?",
        type_resultat="texte",
        k=5,
    )
    assert len(text_hits) == 5
    assert all(hit.type == "texte" for hit in text_hits)

    # 2. Recherche image uniquement
    img_hits = retriever.search(
        texte="Scaled Dot-Product Attention and Multi-Head Attention diagram",
        type_resultat="image",
        k=3,
    )
    assert len(img_hits) >= 1
    assert all(hit.type == "image" for hit in img_hits)

    # 3. Recherche par image en entrée (Figure 1 page 3)
    page3_images = sorted((tmp_path / "extract" / "images").glob("page_003_*.png"))
    assert page3_images
    visual_hits = retriever.search(image=page3_images[0], type_resultat="image", k=2)
    assert len(visual_hits) >= 1
    assert visual_hits[0].page == 3
    assert visual_hits[0].similarity > 0.95

    # 4. Recherche hybride compensant le modality gap CLIP
    hybrid_hits = retriever.search(
        texte="Transformer encoder decoder architecture",
        type_resultat="hybride",
        k=6,
    )
    assert len(hybrid_hits) == 6
    assert any(hit.type == "texte" for hit in hybrid_hits)
    assert any(hit.type == "image" for hit in hybrid_hits)
