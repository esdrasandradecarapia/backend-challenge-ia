from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

# check_same_thread=False é necessário porque o FastAPI executa endpoints
# síncronos em um pool de threads, e o SQLite, por padrão, só aceita uso
# da conexão na thread que a criou.
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    """Dependência do FastAPI: uma sessão por requisição, sempre fechada ao final."""
    with SessionLocal() as db:
        yield db
