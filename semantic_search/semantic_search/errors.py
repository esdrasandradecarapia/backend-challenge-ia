"""Erros esperados do domínio. A CLI os mostra ao usuário sem traceback."""


class SemanticSearchError(Exception):
    """Base de todos os erros esperados deste projeto."""


class DocumentLoadError(SemanticSearchError):
    """Pasta ou arquivo de documento ausente, ilegível ou fora do formato."""


class IndexNotFoundError(SemanticSearchError):
    """O índice ainda não foi criado."""


class IndexCorruptedError(SemanticSearchError):
    """Os arquivos do índice existem, mas estão ilegíveis ou inconsistentes."""


class IndexMismatchError(SemanticSearchError):
    """O índice foi criado com um modelo de embeddings diferente do atual."""


class InvalidQueryError(SemanticSearchError):
    """Consulta vazia."""
