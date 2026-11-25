"""Tests unitaires pour l'extraction PDF (Étape 1) et le découpage BPE CLIP."""

from __future__ import annotations

from pathlib import Path

import pytest

from multimodal_rag.chunking import CLIPTextChunker
from multimodal_rag.config import PROJECT_ROOT
from multimodal_rag.extraction import (
    compute_document_id,
    extract_page_number,
    extract_pdf_content,
)

PDF_PATH = PROJECT_ROOT / "attention is all you need.pdf"


def test_compute_document_id_deterministic() -> None:
    doc_id_1 = compute_document_id(PDF_PATH)
    doc_id_2 = compute_document_id(PDF_PATH)
    assert doc_id_1 == doc_id_2
    assert len(doc_id_1) == 24
    assert all(c in "0123456789abcdef" for c in doc_id_1)


def test_extract_page_number() -> None:
    assert extract_page_number("page_003_image_94.png") == 3
    assert extract_page_number(Path("page_015.txt")) == 15
    assert extract_page_number("architecture_transformer.png") == -1


def test_extract_pdf_content(tmp_path: Path) -> None:
    result = extract_pdf_content(PDF_PATH, tmp_path / "resultats_pdf")

    assert result.total_pages == 11
    assert len(result.pages) == 11
    assert result.full_text_path.is_file()
    full_text = result.full_text_path.read_text(encoding="utf-8")
    assert "Attention Is All You Need" in full_text
    assert len(result.pages_without_text) == 0

    assert len(result.images) >= 3
    page3_imgs = [img for img in result.images if img.page_number == 3]
    page4_imgs = [img for img in result.images if img.page_number == 4]
    assert len(page3_imgs) >= 1
    assert len(page4_imgs) >= 1
    assert all(img.path.is_file() for img in result.images)


def test_clip_chunker_validation() -> None:
    with pytest.raises(ValueError):
        CLIPTextChunker(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        CLIPTextChunker(chunk_size=80, chunk_overlap=10)


def test_clip_chunker_respects_bpe_limit() -> None:
    chunker = CLIPTextChunker(chunk_size=60, chunk_overlap=10)

    sample_text = (
        "The Transformer allows for significantly more parallelization and can reach "
        "a new state of the art in translation quality after being trained for as "
        "little as twelve hours on eight P100 GPUs. " * 12
    )
    chunks = chunker.split_text(sample_text)
    assert len(chunks) >= 3
    assert all(chunker.token_count(chunk) <= 60 for chunk in chunks)

    # Texte synthétique sans espaces pour tester le fallback BPE strict.
    spaceless = "transformerattentionmechanism" * 30
    spaceless_chunks = chunker.split_text(spaceless)
    assert len(spaceless_chunks) >= 2
    assert all(chunker.token_count(chunk) <= 60 for chunk in spaceless_chunks)
