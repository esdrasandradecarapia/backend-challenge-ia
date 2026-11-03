"""Armazenamento e busca de vetores com FAISS."""

from pathlib import Path
from typing import NamedTuple

import faiss
import numpy as np

_UNIT_NORM_TOLERANCE = 1e-3


class Match(NamedTuple):
    position: int  # posição do vetor, na ordem em que foi adicionado
    score: float  # produto interno com a consulta (= cosseno, se normalizados)


class FaissVectorStore:
    """Índice FAISS de busca exata (IndexFlatIP).

    O IndexFlatIP compara a consulta com todos os vetores e ordena pelo produto
    interno. Com vetores normalizados, isso é a similaridade de cosseno. Para um
    corpus pequeno a busca exata é a mais simples e não perde resultado; com
    milhões de vetores, valeria um índice aproximado (HNSW, IVF).
    """

    def __init__(self, index: faiss.Index) -> None:
        self._index = index

    @classmethod
    def empty(cls, dimension: int) -> "FaissVectorStore":
        return cls(faiss.IndexFlatIP(dimension))

    @property
    def dimension(self) -> int:
        return self._index.d

    def __len__(self) -> int:
        return self._index.ntotal

    def add(self, vectors: np.ndarray) -> None:
        """Adiciona vetores float32 de formato (n, dimensão) e norma 1."""
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"Esperado formato (n, {self.dimension}), recebido {vectors.shape}.")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=_UNIT_NORM_TOLERANCE):
            raise ValueError("Os vetores precisam estar normalizados (norma 1).")
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, query: np.ndarray, top_k: int) -> list[Match]:
        """Devolve até top_k vetores mais similares, do mais para o menos similar."""
        if query.shape != (self.dimension,):
            raise ValueError(
                f"Esperado vetor de formato ({self.dimension},), recebido {query.shape}."
            )
        k = min(top_k, len(self))
        if k == 0:
            return []

        scores, positions = self._index.search(
            np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32), k
        )
        return [Match(int(p), float(s)) for p, s in zip(positions[0], scores[0], strict=True)]

    def save(self, path: Path) -> None:
        # Serializar para bytes e gravar com Python evita um problema conhecido do
        # FAISS no Windows com caminhos que têm acentos.
        path.write_bytes(faiss.serialize_index(self._index).tobytes())

    @classmethod
    def load(cls, path: Path) -> "FaissVectorStore":
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        return cls(faiss.deserialize_index(data))
