"""Étape 2 : Vectorisation multimodale texte & image dans un espace latent CLIP."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import open_clip
import torch
from PIL import Image


class CLIPEmbedder:
    """Encodeur multimodal CLIP produisant des embeddings L2-normalisés."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self._embedding_dim = int(self.model.text_projection.shape[1])

    @property
    def embedding_dim(self) -> int:
        """Dimension des vecteurs de l'espace latent commun."""
        return self._embedding_dim

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode et normalise L2 une liste de textes."""
        if not texts:
            return []
        tokens = self.tokenizer(texts).to(self.device)
        vectors = self.model.encode_text(tokens).float()
        vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return vectors.cpu().tolist()

    @torch.inference_mode()
    def encode_image(self, image_path: str | Path) -> list[float]:
        """Encode et normalise L2 une image locale."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image introuvable : {path.resolve()}")

        with Image.open(path) as img:
            tensor = self.preprocess(img.convert("RGB"))

        batch = tensor.unsqueeze(0).to(self.device)
        vector = self.model.encode_image(batch).float()
        vector = vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return vector.cpu().tolist()[0]

    @torch.inference_mode()
    def encode_images(self, image_paths: list[Path]) -> list[list[float]]:
        """Encode et normalise L2 un lot d'images locales."""
        if not image_paths:
            return []
        tensors = []
        for image_path in image_paths:
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError(f"Image introuvable : {path.resolve()}")
            with Image.open(path) as img:
                tensors.append(self.preprocess(img.convert("RGB")))

        batch = torch.stack(tensors).to(self.device)
        vectors = self.model.encode_image(batch).float()
        vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return vectors.cpu().tolist()
