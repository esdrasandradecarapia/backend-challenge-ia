"""Teste com o modelo REAL de embeddings. Não roda por padrão.

Baixa o modelo do Hugging Face (cerca de 470 MB, só na primeira vez) e confere se a
busca semântica encontra o documento certo mesmo quando a consulta não usa as mesmas
palavras do texto. Para rodar:

    python -m pytest -m integration -v
"""

from pathlib import Path

import pytest

from semantic_search.config import Settings
from semantic_search.engine import SemanticSearchEngine
from semantic_search.loader import load_documents
from semantic_search.transformers_embedder import TransformersEmbedder

pytestmark = pytest.mark.integration

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"

# (consulta, documento esperado). A consulta evita de propósito as palavras-chave do texto.
EXPECTED = [
    ("fazer meu programa funcionar igual em qualquer computador", "containers-docker"),
    ("deixar as consultas ao banco de dados mais rápidas", "indices-banco-dados"),
    (
        "impedir que um atacante execute comandos maliciosos pelo formulário do site",
        "seguranca-sql-injection",
    ),
    ("descobrir por que o sistema está lento em produção", "observabilidade"),
    (
        "ensinar o computador a reconhecer padrões a partir de exemplos",
        "aprendizado-supervisionado",
    ),
    ("garantir que cada mudança no código seja verificada antes de ir ao ar", "ci-cd"),
    ("um assistente que responde perguntas usando os documentos da minha empresa", "llms-rag"),
]


@pytest.fixture(scope="module")
def engine():
    settings = Settings()
    embedder = TransformersEmbedder(
        settings.model_name,
        query_prefix=settings.query_prefix,
        document_prefix=settings.document_prefix,
    )
    return SemanticSearchEngine.build(embedder, load_documents(DOCUMENTS_DIR))


@pytest.mark.parametrize(("query", "expected_id"), EXPECTED)
def test_expected_document_is_among_the_top_three(engine, query, expected_id):
    results = engine.search(query, top_k=3)

    ranking = [(r.document.id, round(r.score, 3)) for r in results]
    assert expected_id in [r.document.id for r in results], f"obtido: {ranking}"
