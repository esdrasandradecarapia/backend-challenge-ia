from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class BookCreate(BaseModel):
    """Dados necessários para cadastrar um livro. Todos os campos são obrigatórios."""

    # Remove espaços nas pontas antes de validar, assim "   " é rejeitado
    # pelo min_length e "  Dom Casmurro " é salvo como "Dom Casmurro".
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "title": "Dom Casmurro",
                    "author": "Machado de Assis",
                    "published_date": "1899-01-01",
                    "summary": "Bentinho narra a sua vida e a suspeita de traição de Capitu.",
                }
            ]
        },
    )

    title: str = Field(min_length=1, max_length=255, description="Título do livro")
    author: str = Field(min_length=1, max_length=255, description="Autor do livro")
    published_date: date = Field(description="Data de publicação (formato AAAA-MM-DD)")
    summary: str = Field(min_length=1, max_length=5000, description="Resumo do livro")


class BookRead(BookCreate):
    """Livro como retornado pela API (inclui o id gerado)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
