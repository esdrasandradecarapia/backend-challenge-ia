"""Motor de busca semântica: junta o embedder, o vector store e os documentos."""

from collections.abc import Sequence

from semantic_search.embedder import Embedder
from semantic_search.errors import InvalidQueryError
from semantic_search.models import Document, SearchResult
from semantic_search.vector_store import FaissVectorStore


class SemanticSearchEngine:
    """O vetor na posição i do store representa o documento i da lista."""

    def __init__(
        self,
        embedder: Embedder,
        store: FaissVectorStore,
        documents: Sequence[Document],
    ) -> None:
        if len(store) != len(documents):
            raise ValueError(
                f"O store tem {len(store)} vetores, mas há {len(documents)} documentos."
            )
        self._embedder = embedder
        self._store = store
        self._documents = list(documents)

    @classmethod
    def build(cls, embedder: Embedder, documents: Sequence[Document]) -> "SemanticSearchEngine":
        """Gera os embeddings dos documentos e monta um índice novo."""
        if not documents:
            raise ValueError("É preciso ao menos um documento para criar o índice.")

        vectors = embedder.embed_documents([document.searchable_text for document in documents])
        store = FaissVectorStore.empty(dimension=vectors.shape[1])
        store.add(vectors)
        return cls(embedder, store, documents)

    @property
    def store(self) -> FaissVectorStore:
        return self._store

    @property
    def documents(self) -> list[Document]:
        return list(self._documents)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Devolve os top_k documentos mais próximos da consulta, do mais relevante ao menos."""
        if top_k < 1:
            raise ValueError("top_k deve ser pelo menos 1")
        query = query.strip()
        if not query:
            raise InvalidQueryError("A consulta não pode estar vazia.")

        matches = self._store.search(self._embedder.embed_query(query), top_k)
        return [SearchResult(self._documents[m.position], m.score) for m in matches]
