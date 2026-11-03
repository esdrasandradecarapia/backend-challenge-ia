"""Contrato dos geradores de embeddings."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Transforma texto em vetores.

    Contrato dos vetores devolvidos: float32 e já normalizados (norma 1). Assim, o
    produto interno entre dois vetores é a similaridade de cosseno entre eles.

    Há dois métodos porque alguns modelos tratam diferente o que é consulta e o que
    é documento (por exemplo, com prefixos distintos).
    """

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Devolve um array de formato (n, dimensão), um vetor por texto."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Devolve um array de formato (dimensão,)."""
        ...
