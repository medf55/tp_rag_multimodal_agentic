"""Configuration centralisée du pipeline RAG multimodal agentique."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Paramètres applicatifs chargés depuis l'environnement et `.env`."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    pdf_path: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "attention is all you need.pdf"
    )
    output_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "resultats_pdf"
    )
    chroma_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "chroma_pdf"
    )
    collection_name: str = "pdf_clip_vit_b32"

    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

    chunk_size: int = 60
    chunk_overlap: int = 10
    batch_size: int = 64

    llm_provider: Literal["gemini", "openai"] = "gemini"
    gemini_model: str = "gemini-3.5-flash-lite"
    openai_model: str = "gpt-4.1-mini"

    max_tool_calls: int = 4
    recursion_limit: int = 20
    max_images_per_generation: int = 3

    def resolve_gemini_api_key(self) -> str:
        """Retourne la clé API Gemini depuis `GOOGLE_API_KEY` ou `GEMINI_KEY`."""
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_KEY")
        if not key or not key.strip():
            raise ValueError(
                "Clé API Gemini manquante. Définissez `GOOGLE_API_KEY` ou `GEMINI_KEY` "
                "dans l'environnement ou dans `.env`."
            )
        return key.strip()

    def resolve_openai_api_key(self) -> str:
        """Retourne la clé API OpenAI depuis `OPENAI_API_KEY`."""
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        key = os.environ.get("OPENAI_API_KEY")
        if not key or not key.strip():
            raise ValueError(
                "Clé API OpenAI manquante. Définissez `OPENAI_API_KEY` "
                "dans l'environnement ou dans `.env`."
            )
        return key.strip()


def get_settings() -> Settings:
    """Charge `.env` puis retourne une instance `Settings`."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings()
