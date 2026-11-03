import pytest

from semantic_search.engine import SemanticSearchEngine
from semantic_search.errors import InvalidQueryError
from semantic_search.models import Document
from semantic_search.vector_store import FaissVectorStore


@pytest.fixture
def engine(embedder, documents):
    return SemanticSearchEngine.build(embedder, documents)


@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("como empacotar uma aplicação em container", "docker"),
        ("deixar a consulta ao banco mais rápida", "banco"),
        ("proteger o site de um ataque malicioso", "seguranca"),
    ],
)
def test_most_relevant_document_comes_first(engine, query, expected_id):
    results = engine.search(query, top_k=3)

    assert results[0].document.id == expected_id


def test_results_are_sorted_from_most_to_least_similar(engine):
    scores = [r.score for r in engine.search("container docker", top_k=3)]

    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_results_and_larger_values_return_all_documents(engine):
    assert len(engine.search("container", top_k=1)) == 1
    assert len(engine.search("container", top_k=99)) == 3


def test_result_carries_the_document_metadata(engine):
    [result] = engine.search("container", top_k=1)

    assert result.document.title == "Docker"
    assert result.document.source == "docker.txt"


def test_blank_query_is_rejected(engine):
    with pytest.raises(InvalidQueryError):
        engine.search("   ")


def test_top_k_must_be_positive(engine):
    with pytest.raises(ValueError, match="top_k"):
        engine.search("container", top_k=0)


def test_building_without_documents_is_rejected(embedder):
    with pytest.raises(ValueError, match="ao menos um documento"):
        SemanticSearchEngine.build(embedder, [])


def test_store_and_documents_must_have_the_same_size(embedder, documents):
    store = FaissVectorStore.empty(dimension=4)

    with pytest.raises(ValueError, match="vetores"):
        SemanticSearchEngine(embedder, store, documents)


def test_documents_property_is_a_copy(engine):
    engine.documents.clear()

    assert len(engine.documents) == 3


def test_title_takes_part_in_the_embedding(embedder):
    """Só o título de "b" menciona o conceito. "b" fica na segunda posição de propósito:
    sem o título os dois documentos empatam, e o desempate por posição devolveria "a".
    """
    documents = [
        Document("a", "Outro assunto", "Texto neutro.", "a.txt"),
        Document("b", "Docker", "Texto neutro.", "b.txt"),
    ]
    engine = SemanticSearchEngine.build(embedder, documents)

    assert engine.search("docker", top_k=1)[0].document.id == "b"
