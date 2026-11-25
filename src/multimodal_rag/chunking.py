"""Découpage du texte aligné sur la fenêtre BPE de CLIP (77 tokens max)."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from open_clip.tokenizer import SimpleTokenizer


class CLIPTextChunker:
    """Découpe un texte en fragments garantis sous la limite BPE de CLIP."""

    MAX_CLIP_TOKENS = 75  # 77 moins les tokens spéciaux SOT et EOT

    def __init__(self, chunk_size: int = 60, chunk_overlap: int = 10) -> None:
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "chunk_overlap doit vérifier 0 <= chunk_overlap < chunk_size."
            )
        if chunk_size > self.MAX_CLIP_TOKENS:
            raise ValueError(
                f"chunk_size ({chunk_size}) doit être <= {self.MAX_CLIP_TOKENS} "
                "pour respecter la fenêtre de 77 tokens de CLIP."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._bpe = SimpleTokenizer()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.token_count,
            separators=["\n\n", "\n", ". ", " ", ""],
            strip_whitespace=True,
        )

    def token_count(self, text: str) -> int:
        """Retourne le nombre de tokens BPE CLIP pour un texte donné."""
        if not text:
            return 0
        return len(self._bpe.encode(text))

    def _split_bpe_strict(self, text: str) -> list[str]:
        """Fallback strict par fenêtre glissante de tokens BPE."""
        tokens = self._bpe.encode(text)
        if not tokens:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []
        for start in range(0, len(tokens), step):
            fragment = self._bpe.decode(tokens[start : start + self.chunk_size]).strip()
            if fragment:
                chunks.append(fragment)
            if start + self.chunk_size >= len(tokens):
                break
        return chunks

    def split_text(self, text: str) -> list[str]:
        """Découpe un texte en respectant les limites naturelles et `chunk_size` BPE."""
        if not text or not text.strip():
            return []

        raw_chunks = self._splitter.split_text(text)
        valid_chunks: list[str] = []

        for chunk in raw_chunks:
            cleaned = chunk.strip()
            if not cleaned:
                continue
            if self.token_count(cleaned) <= self.chunk_size:
                valid_chunks.append(cleaned)
            else:
                # Garantit l'invariant même sur une chaîne sans séparateur (ex. URL).
                for sub_chunk in self._split_bpe_strict(cleaned):
                    if self.token_count(sub_chunk) <= self.chunk_size:
                        valid_chunks.append(sub_chunk)
                    else:
                        # Dernier garde-fou si decode+re-encode ajoute 1 token BPE.
                        tokens = self._bpe.encode(sub_chunk)[: self.chunk_size]
                        trimmed = self._bpe.decode(tokens).strip()
                        if trimmed:
                            valid_chunks.append(trimmed)

        return valid_chunks
