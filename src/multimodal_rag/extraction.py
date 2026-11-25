"""Étape 1 : Extraction multimodale (textes et images) d'un PDF via PyMuPDF."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pymupdf

from multimodal_rag.models import (
    ExtractedImage,
    ExtractedPage,
    ExtractionResult,
)


def compute_document_id(pdf_path: Path, length: int = 24) -> str:
    """Calcule un identifiant déterministe par hachage SHA-256 du fichier PDF."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Fichier PDF introuvable : {pdf_path.resolve()}")
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    return digest[:length]


def extract_page_number(path: Path | str) -> int:
    """Extrait le numéro de page depuis un nom de fichier `page_XXX...`."""
    match = re.search(r"page_(\d+)", Path(path).stem)
    return int(match.group(1)) if match else -1


def extract_pdf_content(pdf_path: Path, output_dir: Path) -> ExtractionResult:
    """Extrait le texte page par page et les images intégrées d'un document PDF."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"Fichier PDF introuvable : {pdf_path.resolve()}")

    text_dir = output_dir / "textes"
    image_dir = output_dir / "images"
    text_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    document_id = compute_document_id(pdf_path)
    extracted_pages: list[ExtractedPage] = []
    extracted_images: list[ExtractedImage] = []
    pages_without_text: list[int] = []
    full_text_blocks: list[str] = []

    with pymupdf.open(pdf_path) as document:
        total_pages = len(document)

        for page_num, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)
            text_file = text_dir / f"page_{page_num:03d}.txt"
            text_file.write_text(text, encoding="utf-8")

            extracted_pages.append(
                ExtractedPage(
                    page_number=page_num,
                    text=text,
                    text_path=text_file.resolve(),
                )
            )
            full_text_blocks.append(f"===== PAGE {page_num} =====\n\n{text}")

            if not text.strip():
                pages_without_text.append(page_num)

            seen_xrefs: set[int] = set()
            for image_info in page.get_images(full=True):
                xref = int(image_info[0])
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    raw_image = document.extract_image(xref)
                    if not raw_image or not raw_image.get("image"):
                        continue

                    ext = str(raw_image.get("ext", "png")).lower()
                    width = int(raw_image.get("width", 0))
                    height = int(raw_image.get("height", 0))
                    image_file = image_dir / f"page_{page_num:03d}_image_{xref}.{ext}"
                    image_file.write_bytes(raw_image["image"])

                    extracted_images.append(
                        ExtractedImage(
                            page_number=page_num,
                            xref=xref,
                            path=image_file.resolve(),
                            width=width,
                            height=height,
                            ext=ext,
                        )
                    )
                except Exception:
                    # Ignore les flux image invalides ou corrompus sans bloquer l'extraction.
                    continue

    full_text_path = (output_dir / "texte_complet.txt").resolve()
    full_text_path.write_text("\n\n".join(full_text_blocks), encoding="utf-8")

    return ExtractionResult(
        pdf_path=pdf_path.resolve(),
        document_id=document_id,
        total_pages=total_pages,
        pages=extracted_pages,
        images=extracted_images,
        full_text_path=full_text_path,
        pages_without_text=pages_without_text,
    )
