"""Regras de acesso a dados dos livros.

Fica separado do router para que o router só cuide de HTTP
(validação, status code) e a lógica de consulta seja testável isoladamente.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Book
from app.schemas import BookCreate


def create_book(db: Session, data: BookCreate) -> Book:
    book = Book(**data.model_dump())
    db.add(book)
    db.commit()
    db.refresh(book)
    return book


def search_books(
    db: Session,
    *,
    title: str | None,
    author: str | None,
    limit: int,
    offset: int,
) -> list[Book]:
    """Busca parcial por título e/ou autor. Filtros informados se combinam com AND."""
    stmt = select(Book)

    # Filtro em branco ("?title=" ou só espaços) é tratado como filtro ausente.
    # autoescape=True faz "%" e "_" digitados pelo usuário serem buscados
    # literalmente, em vez de funcionarem como curingas do LIKE.
    if title and title.strip():
        stmt = stmt.where(Book.title.icontains(title.strip(), autoescape=True))
    if author and author.strip():
        stmt = stmt.where(Book.author.icontains(author.strip(), autoescape=True))

    # Ordenar por id garante paginação estável entre requisições.
    stmt = stmt.order_by(Book.id).limit(limit).offset(offset)
    return list(db.scalars(stmt))


def get_book(db: Session, book_id: int) -> Book | None:
    return db.get(Book, book_id)
