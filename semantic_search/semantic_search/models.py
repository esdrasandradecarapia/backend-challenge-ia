from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str  # nome do arquivo sem extensão
    title: str
    text: str
    source: str  # nome do arquivo de origem

    @property
    def searchable_text(self) -> str:
        """Texto que vira embedding. O título entra junto porque resume o assunto."""
        return f"{self.title}\n{self.text}"

    def preview(self, max_chars: int = 120) -> str:
        flat = " ".join(self.text.split())
        if len(flat) <= max_chars:
            return flat
        return flat[: max_chars - 3].rstrip() + "..."


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float  # similaridade de cosseno com a consulta
