"""Embedder falso e determinístico, para testar sem baixar modelo."""

from collections.abc import Mapping, Sequence

import numpy as np


class KeywordEmbedder:
    """Cada conceito é uma dimensão; o valor é quantas palavras-chave dele aparecem no texto.

    Uma dimensão extra constante evita vetores nulos, que não podem ser normalizados.
    """

    def __init__(self, concepts: Mapping[str, Sequence[str]]) -> None:
        self._keywords = [tuple(words) for words in concepts.values()]

    def _vectorize(self, text: str) -> np.ndarray:
        lowered = text.lower()
        counts = [sum(lowered.count(word) for word in words) for words in self._keywords]
        vector = np.array([*counts, 0.01], dtype=np.float32)
        return vector / np.linalg.norm(vector)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vectorize(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vectorize(text)
