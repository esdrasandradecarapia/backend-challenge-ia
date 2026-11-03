"""Gravação e leitura do índice no disco.

O índice são dois arquivos na mesma pasta:
  - vectors.faiss: os vetores (FAISS);
  - metadata.json: os documentos, na mesma ordem dos vetores, e o modelo usado.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from semantic_search.errors import IndexCorruptedError, IndexMismatchError, IndexNotFoundError
from semantic_search.models import Document
from semantic_search.vector_store import FaissVectorStore

_FORMAT_VERSION = 1
_REINDEX_HINT = "Rode novamente: python -m semantic_search index"


@dataclass(frozen=True)
class StoredIndex:
    store: FaissVectorStore
    documents: list[Document]


class IndexRepository:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._vectors_path = directory / "vectors.faiss"
        self._metadata_path = directory / "metadata.json"

    def save(self, store: FaissVectorStore, documents: list[Document], model_name: str) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        store.save(self._vectors_path)
        metadata = {
            "format_version": _FORMAT_VERSION,
            "model_name": model_name,
            "dimension": store.dimension,
            "documents": [asdict(document) for document in documents],
        }
        self._metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self, expected_model_name: str) -> StoredIndex:
        if not (self._vectors_path.is_file() and self._metadata_path.is_file()):
            raise IndexNotFoundError(
                f"Índice não encontrado em '{self._directory}'. Crie-o com: "
                "python -m semantic_search index"
            )

        metadata = self._read_metadata()
        if metadata["model_name"] != expected_model_name:
            # Vetores de modelos diferentes vivem em espaços diferentes: compará-los
            # daria resultados sem sentido, mesmo que a dimensão coincida.
            raise IndexMismatchError(
                f"O índice foi criado com o modelo '{metadata['model_name']}', mas o modelo "
                f"atual é '{expected_model_name}'. {_REINDEX_HINT}"
            )

        store = self._read_vectors()
        documents = self._parse_documents(metadata)
        if len(store) != len(documents) or store.dimension != metadata["dimension"]:
            raise IndexCorruptedError(
                f"Os vetores e os documentos do índice não correspondem. {_REINDEX_HINT}"
            )
        return StoredIndex(store=store, documents=documents)

    def _read_metadata(self) -> dict[str, Any]:
        try:
            metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            if metadata["format_version"] != _FORMAT_VERSION:
                raise ValueError(f"versão de formato {metadata['format_version']!r}")
            metadata["model_name"], metadata["dimension"], metadata["documents"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise IndexCorruptedError(
                f"Não foi possível ler {self._metadata_path.name} ({exc}). {_REINDEX_HINT}"
            ) from exc
        return metadata

    def _read_vectors(self) -> FaissVectorStore:
        try:
            return FaissVectorStore.load(self._vectors_path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise IndexCorruptedError(
                f"Não foi possível ler {self._vectors_path.name}. {_REINDEX_HINT}"
            ) from exc

    @staticmethod
    def _parse_documents(metadata: dict[str, Any]) -> list[Document]:
        try:
            return [Document(**item) for item in metadata["documents"]]
        except TypeError as exc:
            raise IndexCorruptedError(
                f"Os documentos gravados no índice estão incompletos. {_REINDEX_HINT}"
            ) from exc
