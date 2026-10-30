# Q1 — API da Biblioteca Virtual

API REST para cadastrar e consultar livros por título ou autor.

## Tecnologias

- Python 3.10+ (testado em 3.12)
- FastAPI (documentação OpenAPI/Swagger automática)
- SQLAlchemy 2.x + SQLite
- Pydantic (validação) e pydantic-settings (configuração por variáveis de ambiente)
- Pytest + TestClient do FastAPI

## Arquitetura

```
app/
├── main.py          # cria o app, registra rotas e cria as tabelas na inicialização
├── config.py        # configurações (DATABASE_URL) via variável de ambiente / .env
├── database.py      # engine, sessão e dependência get_db
├── models.py        # modelo SQLAlchemy (tabela books)
├── schemas.py       # contratos de entrada/saída e validações (Pydantic)
├── services.py      # regras de acesso a dados (criar, buscar, obter)
└── routers/books.py # endpoints HTTP
tests/
├── conftest.py      # cliente de teste com SQLite em memória isolado por teste
└── test_books.py
```

Fluxo: `router` (HTTP) → `services` (consulta/persistência) → `SQLAlchemy Session` → SQLite.
O router só cuida de HTTP; a lógica de busca fica em `services.py`.

## Instalação

```bash
cd q1-books-api

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
# python -m venv .venv
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Configuração

Opcional. Sem configurar nada, o banco será o arquivo `books.db` na pasta atual.

```bash
cp .env.example .env   # e edite DATABASE_URL se quiser outro caminho
```

| Variável       | Padrão                   | Descrição                |
|----------------|--------------------------|--------------------------|
| `DATABASE_URL` | `sqlite:///./books.db`   | URL do banco SQLite      |

## Como executar

```bash
uvicorn app.main:app --reload
```

- API: http://127.0.0.1:8000
- Swagger UI (interativo): http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

As tabelas são criadas automaticamente na primeira execução.

## Como executar os testes

```bash
python -m pytest -v
```

Os testes usam um SQLite em memória, então não alteram o `books.db`.

## Endpoints

| Método | Rota                | Descrição                                   | Status       |
|--------|---------------------|---------------------------------------------|--------------|
| POST   | `/books`            | Cadastra um livro                           | 201, 422     |
| GET    | `/books`            | Lista/busca por `title` e/ou `author`       | 200, 422     |
| GET    | `/books/{book_id}`  | Obtém um livro pelo id                      | 200, 404, 422|

Parâmetros de `GET /books`:

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `title`   | texto | Trecho do título. Opcional. |
| `author`  | texto | Trecho do autor. Opcional. |
| `limit`   | int (1–100) | Máximo de resultados. Padrão 20. |
| `offset`  | int (≥ 0) | Quantidade a pular. Padrão 0. |

Se `title` e `author` vierem juntos, o livro precisa atender aos dois (AND).
Filtro em branco é ignorado. Resultados ordenados por id.

## Exemplos de uso

Cadastrar:

```bash
curl -X POST http://127.0.0.1:8000/books \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Dom Casmurro",
    "author": "Machado de Assis",
    "published_date": "1899-01-01",
    "summary": "Bentinho narra a sua vida e a suspeita de traição de Capitu."
  }'
```

Resposta (201):

```json
{
  "title": "Dom Casmurro",
  "author": "Machado de Assis",
  "published_date": "1899-01-01",
  "summary": "Bentinho narra a sua vida e a suspeita de traição de Capitu.",
  "id": 1
}
```

Consultar:

```bash
curl "http://127.0.0.1:8000/books?title=casmurro"
curl "http://127.0.0.1:8000/books?author=machado"
curl "http://127.0.0.1:8000/books?title=dom&author=machado"
curl "http://127.0.0.1:8000/books/1"
```

## Decisões de projeto

- **Sem camada de repository:** o service usa a `Session` do SQLAlchemy direto. Para um CRUD
  pequeno, uma interface extra seria abstração sem ganho real.
- **`create_all` em vez de migrations:** suficiente para este escopo. Se o schema passar a
  evoluir, o próximo passo é o Alembic.
- **Todos os campos obrigatórios:** o enunciado lista os quatro campos sem marcar nenhum
  como opcional. Textos são aparados (`strip`) e não podem ficar em branco.
- **`%` e `_` na busca são literais:** a busca escapa os curingas do `LIKE`, para que
  `?title=%` não retorne o banco inteiro.
- **`GET /books/{id}`:** não estava no enunciado; foi incluído por custar quase nada.
- **`check_same_thread=False`:** necessário porque o FastAPI executa endpoints síncronos em
  um pool de threads e o SQLite, por padrão, restringe a conexão à thread que a criou.

## Limitações conhecidas

- **Busca e acentuação:** o SQLite só ignora maiúsculas/minúsculas em letras ASCII, e a busca
  não ignora acentos. Exemplos reais com o título "Memórias Póstumas":
  `memórias` encontra; `MEMÓRIAS` e `memorias` **não** encontram.
  Corrigir exigiria guardar uma coluna normalizada (sem acentos) para busca,
  o que foi deixado de fora por estar além do escopo pedido.
- **`LIKE '%texto%'` não usa índice:** aceitável para uma biblioteca pequena; em volume grande
  seria necessário FTS5 ou outro mecanismo de busca.
- Não há autenticação nem edição/remoção de livros, pois não foram pedidas.

## Autor
Esdras Andrade Carapía 2026
