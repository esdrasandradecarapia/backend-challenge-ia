import pytest

from semantic_search.models import Document
from tests.fakes import KeywordEmbedder

CONCEPTS = {
    "containers": ["container", "docker", "imagem"],
    "banco": ["banco", "índice", "consulta", "sql"],
    "seguranca": ["ataque", "injeção", "malicios"],
}


@pytest.fixture
def embedder():
    return KeywordEmbedder(CONCEPTS)


@pytest.fixture
def documents():
    return [
        Document("docker", "Docker", "Um container empacota a imagem da aplicação.", "docker.txt"),
        Document(
            "banco", "Banco de dados", "Índices aceleram cada consulta ao banco.", "banco.txt"
        ),
        Document(
            "seguranca",
            "Segurança",
            "Uma injeção de SQL é um ataque malicioso.",
            "seguranca.txt",
        ),
    ]
