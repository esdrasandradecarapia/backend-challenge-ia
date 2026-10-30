import pytest

VALID_BOOK = {
    "title": "Dom Casmurro",
    "author": "Machado de Assis",
    "published_date": "1899-01-01",
    "summary": "Bentinho narra a sua vida e a suspeita de traição de Capitu.",
}


def create_book(client, **overrides):
    """Cadastra um livro (com campos sobrescritos) e devolve o JSON da resposta."""
    response = client.post("/books", json={**VALID_BOOK, **overrides})
    assert response.status_code == 201, response.text
    return response.json()


# ---------- Cadastro ----------


def test_create_book_returns_201_with_generated_id(client):
    response = client.post("/books", json=VALID_BOOK)

    assert response.status_code == 201
    assert response.json() == {"id": 1, **VALID_BOOK}


def test_create_book_is_persisted(client):
    created = create_book(client)

    response = client.get(f"/books/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


@pytest.mark.parametrize("missing_field", list(VALID_BOOK))
def test_create_book_requires_every_field(client, missing_field):
    payload = {k: v for k, v in VALID_BOOK.items() if k != missing_field}

    response = client.post("/books", json=payload)

    assert response.status_code == 422


def test_create_book_rejects_invalid_date(client):
    response = client.post("/books", json={**VALID_BOOK, "published_date": "31/12/1899"})

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["title", "author", "summary"])
@pytest.mark.parametrize("blank", ["", "   "])
def test_create_book_rejects_blank_text_fields(client, field, blank):
    response = client.post("/books", json={**VALID_BOOK, field: blank})

    assert response.status_code == 422


def test_create_book_trims_surrounding_whitespace(client):
    created = create_book(client, title="  Dom Casmurro  ")

    assert created["title"] == "Dom Casmurro"


def test_create_book_rejects_title_longer_than_limit(client):
    response = client.post("/books", json={**VALID_BOOK, "title": "x" * 256})

    assert response.status_code == 422


# ---------- Consulta por título / autor ----------


@pytest.fixture
def library(client):
    """Três livros: dois do mesmo autor e um de outro."""
    create_book(client, title="Dom Casmurro", author="Machado de Assis")
    create_book(client, title="Memórias Póstumas de Brás Cubas", author="Machado de Assis")
    create_book(client, title="Grande Sertão: Veredas", author="João Guimarães Rosa")
    return client


def titles(response):
    return [book["title"] for book in response.json()]


def test_list_without_filters_returns_all_books(library):
    response = library.get("/books")

    assert response.status_code == 200
    assert len(response.json()) == 3


def test_list_on_empty_database_returns_empty_list(client):
    response = client.get("/books")

    assert response.status_code == 200
    assert response.json() == []


def test_search_by_title_matches_partial_text_ignoring_case(library):
    response = library.get("/books", params={"title": "casmurro"})

    assert response.status_code == 200
    assert titles(response) == ["Dom Casmurro"]


def test_search_by_author_returns_all_books_of_that_author(library):
    response = library.get("/books", params={"author": "MACHADO"})

    assert response.status_code == 200
    assert titles(response) == ["Dom Casmurro", "Memórias Póstumas de Brás Cubas"]


def test_search_with_title_and_author_applies_both_filters(library):
    response = library.get("/books", params={"title": "dom", "author": "machado"})
    assert titles(response) == ["Dom Casmurro"]

    # Título existe, mas é de outro autor: o AND deve excluir.
    response = library.get("/books", params={"title": "dom", "author": "rosa"})
    assert response.json() == []


def test_search_without_match_returns_empty_list(library):
    response = library.get("/books", params={"title": "inexistente"})

    assert response.status_code == 200
    assert response.json() == []


def test_blank_filter_is_ignored(library):
    response = library.get("/books", params={"title": "   "})

    assert len(response.json()) == 3


def test_like_wildcards_in_search_are_treated_literally(library):
    # Sem escape, "%" casaria com qualquer título.
    response = library.get("/books", params={"title": "%"})

    assert response.json() == []


# ---------- Paginação ----------


def test_pagination_with_limit_and_offset(library):
    first_page = library.get("/books", params={"limit": 2, "offset": 0})
    second_page = library.get("/books", params={"limit": 2, "offset": 2})

    assert titles(first_page) == ["Dom Casmurro", "Memórias Póstumas de Brás Cubas"]
    assert titles(second_page) == ["Grande Sertão: Veredas"]


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"limit": "abc"}],
)
def test_invalid_pagination_params_return_422(client, params):
    response = client.get("/books", params=params)

    assert response.status_code == 422


# ---------- Consulta por id ----------


def test_get_unknown_book_returns_404(client):
    response = client.get("/books/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Livro não encontrado"}


def test_get_book_with_non_numeric_id_returns_422(client):
    response = client.get("/books/abc")

    assert response.status_code == 422
