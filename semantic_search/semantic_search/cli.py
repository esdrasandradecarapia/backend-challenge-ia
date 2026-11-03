"""Interface de linha de comando: `index` cria o índice e `search` faz uma consulta."""

import argparse
import sys
from collections.abc import Callable, Sequence

from semantic_search.config import Settings, load_settings
from semantic_search.embedder import Embedder
from semantic_search.engine import SemanticSearchEngine
from semantic_search.errors import SemanticSearchError
from semantic_search.index_repository import IndexRepository
from semantic_search.loader import load_documents
from semantic_search.models import SearchResult

DEFAULT_TOP_K = 3

EmbedderFactory = Callable[[Settings], Embedder]


def create_embedder(settings: Settings) -> Embedder:
    # Import tardio: o PyTorch demora para carregar e só é necessário aqui,
    # não para --help nem para erros de uso.
    from semantic_search.transformers_embedder import TransformersEmbedder

    print(
        f"Carregando o modelo '{settings.model_name}' "
        "(na primeira vez, o download pode levar alguns minutos)...",
        file=sys.stderr,
    )
    return TransformersEmbedder(
        settings.model_name,
        query_prefix=settings.query_prefix,
        document_prefix=settings.document_prefix,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    embedder_factory: EmbedderFactory = create_embedder,
) -> int:
    args = _build_parser().parse_args(argv)
    settings = load_settings()

    try:
        if args.command == "index":
            _run_index(settings, embedder_factory)
        else:
            _run_search(settings, embedder_factory, args.query, args.top_k)
    except SemanticSearchError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_index(settings: Settings, embedder_factory: EmbedderFactory) -> None:
    documents = load_documents(settings.documents_dir)
    engine = SemanticSearchEngine.build(embedder_factory(settings), documents)
    IndexRepository(settings.index_dir).save(engine.store, engine.documents, settings.model_name)

    print(f"{len(documents)} documentos indexados (vetores de {engine.store.dimension} dimensões).")
    print(f"Índice salvo em: {settings.index_dir}")


def _run_search(
    settings: Settings,
    embedder_factory: EmbedderFactory,
    query: str,
    top_k: int,
) -> None:
    # O índice é lido antes do modelo: se ele não existir, o erro aparece na hora,
    # sem esperar o carregamento do modelo.
    stored = IndexRepository(settings.index_dir).load(expected_model_name=settings.model_name)
    engine = SemanticSearchEngine(embedder_factory(settings), stored.store, stored.documents)
    _print_results(query, engine.search(query, top_k))


def _print_results(query: str, results: list[SearchResult]) -> None:
    print(f"\nConsulta: {query}\n")
    for position, result in enumerate(results, start=1):
        document = result.document
        print(f"{position}. {document.title}  (similaridade {result.score:.3f})")
        print(f"   {document.source}")
        print(f"   {document.preview()}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m semantic_search",
        description="Busca semântica em documentos de texto.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("index", help="lê os documentos, gera os embeddings e salva o índice")

    search = commands.add_parser("search", help="busca os documentos mais relevantes para um texto")
    search.add_argument("query", type=_non_blank_text, help="texto da consulta")
    search.add_argument(
        "--top-k",
        type=_positive_int,
        default=DEFAULT_TOP_K,
        help=f"quantos resultados mostrar (padrão: {DEFAULT_TOP_K})",
    )
    return parser


def _non_blank_text(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("a consulta não pode estar vazia")
    return value


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' não é um número inteiro") from None
    if number < 1:
        raise argparse.ArgumentTypeError("precisa ser pelo menos 1")
    return number
