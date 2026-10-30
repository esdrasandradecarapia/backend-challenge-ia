from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app import services
from app.database import get_db
from app.schemas import BookCreate, BookRead

router = APIRouter(prefix="/books", tags=["Livros"])

DbSession = Annotated[Session, Depends(get_db)]


@router.post(
    "",
    response_model=BookRead,
    status_code=status.HTTP_201_CREATED,
    summary="Cadastrar um livro",
)
def create_book(payload: BookCreate, db: DbSession):
    """Cadastra um livro. Título, autor, data de publicação e resumo são obrigatórios."""
    return services.create_book(db, payload)


@router.get(
    "",
    response_model=list[BookRead],
    summary="Consultar livros por título ou autor",
)
def list_books(
    db: DbSession,
    title: Annotated[
        str | None,
        Query(description="Trecho do título (sem diferenciar maiúsculas de minúsculas)"),
    ] = None,
    author: Annotated[
        str | None,
        Query(description="Trecho do nome do autor (sem diferenciar maiúsculas de minúsculas)"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Máximo de livros retornados")] = 20,
    offset: Annotated[int, Query(ge=0, description="Quantidade de livros a pular")] = 0,
):
    """Lista livros. Sem filtros, retorna todos (paginados).

    Se `title` e `author` forem informados juntos, o livro precisa atender aos dois.
    """
    return services.search_books(db, title=title, author=author, limit=limit, offset=offset)


@router.get(
    "/{book_id}",
    response_model=BookRead,
    summary="Obter um livro pelo id",
    responses={404: {"description": "Livro não encontrado"}},
)
def get_book(book_id: int, db: DbSession):
    book = services.get_book(db, book_id)
    if book is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Livro não encontrado")
    return book
