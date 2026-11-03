"""Leitura dos documentos de texto.

Formato de cada arquivo .txt: a primeira linha é o título; o resto é o texto.
"""

from pathlib import Path

from semantic_search.errors import DocumentLoadError
from semantic_search.models import Document


def load_documents(directory: Path) -> list[Document]:
    if not directory.is_dir():
        raise DocumentLoadError(f"Pasta de documentos não encontrada: {directory}")

    paths = sorted(directory.glob("*.txt"))
    if not paths:
        raise DocumentLoadError(f"Nenhum arquivo .txt encontrado em: {directory}")

    return [_read_document(path) for path in paths]


def _read_document(path: Path) -> Document:
    try:
        # utf-8-sig aceita arquivos salvos com BOM, comum no Bloco de Notas do Windows.
        content = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise DocumentLoadError(f"Não foi possível ler {path.name}: {exc}") from exc

    title, _, body = content.strip().partition("\n")
    title, body = title.strip(), body.strip()
    if not title or not body:
        raise DocumentLoadError(
            f"{path.name}: o arquivo precisa ter o título na primeira linha e o texto depois dela."
        )

    return Document(id=path.stem, title=title, text=body, source=path.name)
