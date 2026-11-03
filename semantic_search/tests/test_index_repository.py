import json

import pytest

from semantic_search.engine import SemanticSearchEngine
from semantic_search.errors import IndexCorruptedError, IndexMismatchError, IndexNotFoundError
from semantic_search.index_repository import IndexRepository

MODEL = "modelo-teste"


@pytest.fixture
def engine(embedder, documents):
    return SemanticSearchEngine.build(embedder, documents)


@pytest.fixture
def repository(tmp_path):
    return IndexRepository(tmp_path / "index")


@pytest.fixture
def saved(repository, engine):
    repository.save(engine.store, engine.documents, MODEL)
    return repository


def test_saved_index_can_be_loaded_with_documents_and_vectors(saved, engine, embedder):
    stored = saved.load(expected_model_name=MODEL)

    restored = SemanticSearchEngine(embedder, stored.store, stored.documents)
    assert stored.documents == engine.documents
    assert restored.search("container docker", top_k=3) == engine.search(
        "container docker", top_k=3
    )


def test_accented_text_survives_the_round_trip(saved):
    stored = saved.load(expected_model_name=MODEL)

    assert "Índices aceleram" in stored.documents[1].text


def test_loading_before_saving_says_how_to_create_the_index(repository):
    with pytest.raises(IndexNotFoundError, match="python -m semantic_search index"):
        repository.load(expected_model_name=MODEL)


def test_index_built_with_another_model_is_rejected(saved):
    with pytest.raises(IndexMismatchError, match="outro-modelo"):
        saved.load(expected_model_name="outro-modelo")


def test_invalid_json_is_reported_as_corrupted(saved, tmp_path):
    (tmp_path / "index" / "metadata.json").write_text("{isso não é json", encoding="utf-8")

    with pytest.raises(IndexCorruptedError, match="metadata.json"):
        saved.load(expected_model_name=MODEL)


def test_metadata_missing_fields_is_reported_as_corrupted(saved, tmp_path):
    path = tmp_path / "index" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    del metadata["documents"]
    path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(IndexCorruptedError):
        saved.load(expected_model_name=MODEL)


def test_incomplete_document_entry_is_reported_as_corrupted(saved, tmp_path):
    path = tmp_path / "index" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    del metadata["documents"][0]["title"]
    path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(IndexCorruptedError, match="incompletos"):
        saved.load(expected_model_name=MODEL)


def test_unknown_format_version_is_reported_as_corrupted(saved, tmp_path):
    path = tmp_path / "index" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["format_version"] = 99
    path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(IndexCorruptedError):
        saved.load(expected_model_name=MODEL)


def test_garbage_in_the_vectors_file_is_reported_as_corrupted(saved, tmp_path):
    (tmp_path / "index" / "vectors.faiss").write_bytes(b"isto nao e um indice faiss")

    with pytest.raises(IndexCorruptedError, match="vectors.faiss"):
        saved.load(expected_model_name=MODEL)


def test_vectors_and_documents_out_of_sync_are_reported_as_corrupted(saved, tmp_path):
    path = tmp_path / "index" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["documents"].pop()
    path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(IndexCorruptedError, match="não correspondem"):
        saved.load(expected_model_name=MODEL)
