"""Étape 5 : Outils agentiques encapsulant le retriever multimodal et l'analyse visuelle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool, tool

from multimodal_rag.llm import MultimodalRAGPipeline
from multimodal_rag.retriever import MultimodalRetriever


def create_rag_tools(
    retriever: MultimodalRetriever,
    rag_pipeline: MultimodalRAGPipeline,
) -> list[BaseTool]:
    """Construit les outils LangChain destinés à l'agent ReAct multimodal."""

    @tool
    def retriever_tool(
        requete: str,
        type_resultat: Literal["hybride", "texte", "image", "tous"] = "hybride",
        k: int = 6,
        chemin_image: str | None = None,
    ) -> str:
        """
        Rechercher des passages textuels et/ou des figures dans la base vectorielle PDF (CLIP + ChromaDB).

        Quand utiliser cet outil :
        - Pour toute question portant sur le contenu scientifique, les équations,
          les hyperparamètres, les résultats ou les figures du document PDF indexé.

        Choix du paramètre `type_resultat` :
        - "texte" : explications théoriques, chiffres, tableaux, hyperparamètres.
        - "image" : recherche spécifique de schémas, diagrammes ou figures extraites.
        - "hybride" : mode recommandé par défaut ; retourne à la fois les meilleurs passages
          textuels ET les meilleures figures en compensant le modality gap de CLIP.
        - "tous" : recherche brute globale dans l'espace latent sans équilibrage.

        Paramètre optionnel `chemin_image` :
        - Fournir un chemin local d'image valide lorsque l'utilisateur soumet une image
          et que l'on souhaite retrouver des figures visuellement similaires dans le PDF.

        Retourne un JSON contenant `passages_texte`, `figures_trouvees` (avec leur `id`
        utilisable ensuite par `analyser_figure_pdf`), `sources_transmises` et `avertissements`.
        """
        k_bounded = max(1, min(int(k), 12))
        warnings: list[str] = []

        image_arg: Path | None = None
        if chemin_image and chemin_image.strip():
            candidate = Path(chemin_image.strip())
            if candidate.is_file():
                image_arg = candidate
            else:
                warnings.append(
                    f"Chemin d'image ignoré car introuvable : {chemin_image}"
                )

        try:
            sources = retriever.search(
                texte=requete,
                image=image_arg,
                type_resultat=type_resultat,
                k=k_bounded,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "requete": requete,
                    "mode": type_resultat,
                    "nombre_resultats": 0,
                    "passages_texte": [],
                    "figures_trouvees": [],
                    "sources_transmises": [],
                    "avertissements": warnings + [f"Erreur de recherche : {exc}"],
                },
                ensure_ascii=False,
            )

        passages_texte: list[dict[str, Any]] = []
        figures_trouvees: list[dict[str, Any]] = []
        sources_transmises: list[dict[str, Any]] = []

        for src in sources:
            sources_transmises.append(
                {
                    "reference": src.reference,
                    "id": src.id,
                    "type": src.type,
                    "pdf": src.pdf,
                    "page": src.page,
                    "similarite": src.similarity,
                }
            )
            if src.type == "texte":
                passages_texte.append(
                    {
                        "reference": src.reference,
                        "id": src.id,
                        "pdf": src.pdf,
                        "page": src.page,
                        "similarite": src.similarity,
                        "extrait": src.content,
                    }
                )
            else:
                figures_trouvees.append(
                    {
                        "reference": src.reference,
                        "id": src.id,
                        "pdf": src.pdf,
                        "page": src.page,
                        "similarite": src.similarity,
                        "description_courte": src.content,
                    }
                )

        return json.dumps(
            {
                "requete": requete,
                "mode": type_resultat,
                "nombre_resultats": len(sources),
                "passages_texte": passages_texte,
                "figures_trouvees": figures_trouvees,
                "sources_transmises": sources_transmises,
                "avertissements": warnings,
            },
            ensure_ascii=False,
        )

    @tool
    def analyser_figure_pdf(image_id: str, question: str) -> str:
        """
        Observer et analyser visuellement une figure extraite du PDF à partir de son `image_id` ChromaDB.

        Utiliser cet outil après `retriever_tool` lorsque des `figures_trouvees` ont été retournées
        et qu'il faut lire les étiquettes, blocs, flèches ou détails visuels d'une figure.
        """
        record = retriever.store.get_by_id(image_id)
        if not record:
            return json.dumps(
                {"erreur": f"Identifiant introuvable dans ChromaDB : {image_id}"},
                ensure_ascii=False,
            )

        metadata = record.get("metadata") or {}
        if metadata.get("type") != "image":
            return json.dumps(
                {"erreur": f"L'identifiant {image_id} ne correspond pas à une image."},
                ensure_ascii=False,
            )

        image_path = Path(str(metadata.get("source", "")))
        if not image_path.is_file():
            return json.dumps(
                {"erreur": f"Fichier image introuvable sur disque : {image_path}"},
                ensure_ascii=False,
            )

        analysis = rag_pipeline.analyze_image(
            image_path=image_path,
            metadata=metadata,
            question=question,
        )
        return json.dumps(
            {
                "image_id": image_id,
                "pdf": metadata.get("pdf", "inconnu"),
                "page": metadata.get("page", -1),
                "analyse_visuelle": analysis,
            },
            ensure_ascii=False,
        )

    return [retriever_tool, analyser_figure_pdf]
