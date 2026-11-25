"""Étape 4 : Recherche vectorielle multimodale dans ChromaDB."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from multimodal_rag.models import RetrievedSource
from multimodal_rag.vector_store import MultimodalVectorStore

ResultType = Literal["texte", "image", "hybride", "tous"]


class MultimodalRetriever:
    """Moteur de recherche par similarité cosinus sur l'espace latent CLIP."""

    def __init__(self, store: MultimodalVectorStore) -> None:
        self.store = store

    def _compute_query_vector(
        self,
        texte: str | None = None,
        image: str | Path | None = None,
    ) -> list[float]:
        """Calcule un vecteur requête normalisé à partir d'un texte, d'une image ou des deux."""
        has_text = texte is not None and bool(texte.strip())
        has_image = image is not None

        if not has_text and not has_image:
            raise ValueError("Fournissez au moins un texte non vide ou une image.")

        if has_text and not has_image:
            return self.store.embedder.encode_texts([texte.strip()])[0]

        if has_image and not has_text:
            return self.store.embedder.encode_image(image)  # type: ignore[arg-type]

        # Fusion multimodale pondérée quand texte et image sont fournis ensemble.
        text_vec = torch.tensor(
            self.store.embedder.encode_texts([texte.strip()])[0],
            dtype=torch.float32,
        )
        img_vec = torch.tensor(
            self.store.embedder.encode_image(image),  # type: ignore[arg-type]
            dtype=torch.float32,
        )
        fused = 0.45 * text_vec + 0.55 * img_vec
        fused = fused / fused.norm().clamp_min(1e-12)
        return fused.tolist()

    @staticmethod
    def _to_sources(
        raw: dict,
        start_index: int = 1,
    ) -> list[RetrievedSource]:
        """Convertit une réponse ChromaDB brute en objets `RetrievedSource`."""
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        sources: list[RetrievedSource] = []
        for idx, (item_id, doc, meta, dist) in enumerate(
            zip(ids, docs, metas, dists),
            start=start_index,
        ):
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            sources.append(
                RetrievedSource(
                    reference=f"S{idx}",
                    id=str(item_id),
                    type=meta.get("type", "texte"),
                    pdf=str(meta.get("pdf", "inconnu")),
                    page=int(meta.get("page", -1)),
                    source=str(meta.get("source", "")),
                    content=str(doc),
                    similarity=round(similarity, 4),
                )
            )
        return sources

    def search(
        self,
        texte: str | None = None,
        image: str | Path | None = None,
        type_resultat: ResultType | None = None,
        k: int = 5,
        image_quota: int = 2,
    ) -> list[RetrievedSource]:
        """
        Recherche les k meilleurs éléments par similarité cosinus.

        Le mode `"hybride"` compense le CLIP Modality Gap en garantissant une
        représentation équilibrée d'images et de passages textuels.
        """
        if type_resultat not in (None, "texte", "image", "hybride", "tous"):
            raise ValueError(
                "type_resultat doit être 'texte', 'image', 'hybride', 'tous' ou None."
            )
        if k < 1:
            raise ValueError("k doit être supérieur ou égal à 1.")
        if self.store.count() == 0:
            raise ValueError("La collection vectorielle est vide.")

        query_vec = self._compute_query_vector(texte=texte, image=image)

        if type_resultat in ("texte", "image"):
            raw = self.store.query(
                query_embedding=query_vec,
                n_results=k,
                content_type=type_resultat,
            )
            return self._to_sources(raw)

        if type_resultat in (None, "tous"):
            raw = self.store.query(
                query_embedding=query_vec,
                n_results=k,
                content_type=None,
            )
            return self._to_sources(raw)

        # Mode "hybride" : compense l'écart de scores texte-texte vs texte-image dans CLIP.
        available_images = self.store.count("image")
        available_texts = self.store.count("texte")

        if available_images == 0:
            raw_texts = self.store.query(query_vec, n_results=k, content_type="texte")
            return self._to_sources(raw_texts)

        if available_texts == 0:
            raw_imgs = self.store.query(query_vec, n_results=k, content_type="image")
            return self._to_sources(raw_imgs)

        n_images = min(available_images, max(1, min(image_quota, k // 2 or 1)))
        n_texts = max(0, k - n_images)
        if n_texts > available_texts:
            n_texts = available_texts
            n_images = min(available_images, k - n_texts)

        text_sources = (
            self._to_sources(
                self.store.query(query_vec, n_results=n_texts, content_type="texte")
            )
            if n_texts > 0
            else []
        )
        image_sources = (
            self._to_sources(
                self.store.query(query_vec, n_results=n_images, content_type="image"),
                start_index=len(text_sources) + 1,
            )
            if n_images > 0
            else []
        )

        combined = text_sources + image_sources
        for idx, item in enumerate(combined, start=1):
            item.reference = f"S{idx}"
        return combined
