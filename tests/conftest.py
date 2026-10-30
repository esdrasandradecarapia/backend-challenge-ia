import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app


@pytest.fixture
def client():
    """Cliente de teste com um SQLite em memória novo para cada teste.

    StaticPool mantém a mesma conexão durante todo o teste; sem isso, cada
    conexão nova de um SQLite em memória enxergaria um banco vazio.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_db():
        with TestingSession() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db

    # TestClient SEM "with": o lifespan não roda, então o banco real
    # (books.db) não é tocado durante os testes.
    yield TestClient(app)

    app.dependency_overrides.clear()
    engine.dispose()
