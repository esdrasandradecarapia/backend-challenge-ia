import numpy as np
import pytest

from semantic_search.vector_store import FaissVectorStore


def unit(*values):
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def store():
    """Três vetores nos eixos x, y e z."""
    store = FaissVectorStore.empty(dimension=3)
    store.add(np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)]))
    return store


def test_new_store_reports_dimension_and_size(store):
    assert store.dimension == 3
    assert len(store) == 3


def test_search_orders_by_similarity_and_scores_are_cosines(store):
    query = unit(0.2, 1, 0)

    matches = store.search(query, top_k=3)

    assert [m.position for m in matches] == [1, 0, 2]
    assert matches[0].score == pytest.approx(float(query @ unit(0, 1, 0)), abs=1e-5)
    assert matches[0].score > matches[1].score > matches[2].score


def test_top_k_limits_the_number_of_matches(store):
    assert len(store.search(unit(1, 1, 1), top_k=2)) == 2


def test_top_k_larger_than_the_store_returns_everything(store):
    assert len(store.search(unit(1, 1, 1), top_k=50)) == 3


def test_search_on_empty_store_returns_nothing():
    assert FaissVectorStore.empty(dimension=3).search(unit(1, 0, 0), top_k=5) == []


def test_add_rejects_wrong_dimension(store):
    with pytest.raises(ValueError, match="formato"):
        store.add(np.ones((2, 5), dtype=np.float32) / np.sqrt(5))


def test_add_rejects_vectors_that_are_not_normalized(store):
    with pytest.raises(ValueError, match="normalizados"):
        store.add(np.array([[2, 0, 0]], dtype=np.float32))


def test_search_rejects_wrong_query_shape(store):
    with pytest.raises(ValueError, match="formato"):
        store.search(unit(1, 0), top_k=1)


def test_saved_store_returns_the_same_results_after_loading(store, tmp_path):
    # A pasta com acento e cedilha cobre o caso do Windows que motivou a serialização manual.
    path = tmp_path / "índice-ç" / "vetores.faiss"
    path.parent.mkdir()
    query = unit(0.3, 0.2, 1)

    store.save(path)
    loaded = FaissVectorStore.load(path)

    assert len(loaded) == 3 and loaded.dimension == 3
    assert loaded.search(query, top_k=3) == store.search(query, top_k=3)
