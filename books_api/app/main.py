from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import models  # noqa: F401  (garante que a tabela esteja registrada em Base.metadata)
from app.database import Base, engine
from app.routers import books


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Para este escopo, criar as tabelas na inicialização basta.
    # Se o schema passar a evoluir, o próximo passo é adotar o Alembic.
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Biblioteca Virtual",
    description="API para cadastrar e consultar livros por título ou autor.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(books.router)
