"""Configuração lida de variáveis de ambiente. Todas são opcionais."""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


@dataclass(frozen=True)
class Settings:
    documents_dir: Path = Path("data/documents")
    index_dir: Path = Path("index")
    model_name: str = DEFAULT_MODEL
    # Os modelos da família E5 foram treinados com estes prefixos; sem eles a
    # qualidade da busca cai. Para um modelo que não os usa, defina como vazio.
    query_prefix: str = "query: "
    document_prefix: str = "passage: "


def load_settings() -> Settings:
    defaults = Settings()
    return Settings(
        documents_dir=Path(os.getenv("DOCUMENTS_DIR") or defaults.documents_dir),
        index_dir=Path(os.getenv("INDEX_DIR") or defaults.index_dir),
        model_name=os.getenv("EMBEDDING_MODEL") or defaults.model_name,
        # getenv com padrão (e não "or"): string vazia é um valor válido aqui.
        query_prefix=os.getenv("EMBEDDING_QUERY_PREFIX", defaults.query_prefix),
        document_prefix=os.getenv("EMBEDDING_DOCUMENT_PREFIX", defaults.document_prefix),
    )
