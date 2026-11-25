"""Étape 3 : Indexation et stockage vectoriel persistant dans ChromaDB."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import chromadb

from multimodal_rag.chunking import CLIPTextChunker
from multimodal_rag.embeddings import CLIPEmbedder
from multimodal_rag.models import ExtractionResult, IndexingReport


class MultimodalVectorStore:
    """Gestionnaire ChromaDB pour vecteurs multimodaux (textes + images)."""

    def __init__(
        self,
        chroma_dir: Path,
        collection_name: str,
        embedder: CLIPEmbedder,
    ) -> None:
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = embedder

        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def index_extraction(
        self,
        extraction: ExtractionResult,
        chunker: CLIPTextChunker,
        batch_size: int = 64,
    ) -> IndexingReport:
        """Indexe de manière idempotente les textes et images d'un PDF extrait."""
        doc_id = extraction.document_id
        pdf_name = extraction.pdf_path.name

        # Suppression préalable des anciennes entrées liées à ce SHA-256.
        self.collection.delete(where={"document_id": doc_id})

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for page in extraction.pages:
            chunks = chunker.split_text(page.text)
            for idx, fragment in enumerate(chunks):
                ids.append(f"{doc_id}:texte:{page.page_number}:{idx}")
                documents.append(fragment)
                metadatas.append(
                    {
                        "type": "texte",
                        "document_id": doc_id,
                        "pdf": pdf_name,
                        "page": page.page_number,
                        "fragment": idx,
                        "source": str(page.text_path.resolve()),
                    }
                )

        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            batch_docs = documents[start:end]
            self.collection.upsert(
                ids=ids[start:end],
                documents=batch_docs,
                metadatas=metadatas[start:end],
                embeddings=self.embedder.encode_texts(batch_docs),
            )

        images_indexed = 0
        for img in extraction.images:
            if not img.path.is_file():
                continue
            try:
                vector = self.embedder.encode_image(img.path)
            except Exception:
                continue

            self.collection.upsert(
                ids=[f"{doc_id}:image:{img.path.name}"],
                embeddings=[vector],
                documents=[
                    f"Image extraite du PDF : {img.path.name} (page {img.page_number})"
                ],
                metadatas=[
                    {
                        "type": "image",
                        "document_id": doc_id,
                        "pdf": pdf_name,
                        "page": img.page_number,
                        "filename": img.path.name,
                        "source": str(img.path.resolve()),
                    }
                ],
            )
            images_indexed += 1

        return IndexingReport(
            document_id=doc_id,
            pdf_name=pdf_name,
            text_chunks_indexed=len(documents),
            images_indexed=images_indexed,
            total_collection_items=self.collection.count(),
        )

    def count(self, content_type: Literal["texte", "image"] | None = None) -> int:
        """Retourne le nombre exact d'éléments (total ou filtré par type)."""
        if content_type is None:
            return self.collection.count()
        result = self.collection.get(
            where={"type": content_type},
            include=[],
        )
        return len(result.get("ids", []))

    def get_by_id(self, item_id: str) -> dict[str, Any] | None:
        """Récupère un document et ses métadonnées par identifiant ChromaDB."""
        if not item_id:
            return None
        result = self.collection.get(
            ids=[item_id],
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        if not ids:
            return None
        documents = result.get("documents") or [""]
        metadatas = result.get("metadatas") or [{}]
        return {
            "id": ids[0],
            "document": documents[0],
            "metadata": metadatas[0],
        }

    def query(
        self,
        query_embedding: list[float],
        n_results: int,
        content_type: Literal["texte", "image"] | None = None,
    ) -> dict[str, Any]:
        """Exécute une recherche par similarité cosinus sur ChromaDB."""
        available = self.count(content_type)
        if available == 0 or n_results <= 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, available),
            "include": ["documents", "metadatas", "distances"],
        }
        if content_type is not None:
            kwargs["where"] = {"type": content_type}
        return self.collection.query(**kwargs)
