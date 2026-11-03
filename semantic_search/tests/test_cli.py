import pytest

from semantic_search.cli import main
from tests.conftest import CONCEPTS
from tests.fakes import KeywordEmbedder


def fake_factory(settings):
    return KeywordEmbedder(CONCEPTS)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Pasta de documentos e de índice isoladas, escolhidas por variáveis de ambiente."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "docker.txt").write_text("Docker\nUm container empacota a imagem.", encoding="utf-8")
    (docs / "banco.txt").write_text(
        "Banco\nÍndices aceleram a consulta ao banco.", encoding="utf-8"
    )
    (docs / "seguranca.txt").write_text(
        "Segurança\nUm ataque de injeção malicioso.", encoding="utf-8"
    )

    monkeypatch.setenv("DOCUMENTS_DIR", str(docs))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    return tmp_path


def run(*argv):
    return main(list(argv), embedder_factory=fake_factory)


def test_index_then_search_shows_the_most_relevant_document_first(workspace, capsys):
    assert run("index") == 0
    assert "3 documentos indexados" in capsys.readouterr().out

    assert run("search", "como empacotar em container") == 0

    output = capsys.readouterr().out
    assert output.index("1. Docker") < output.index("2. ")
    assert "docker.txt" in output
    assert "similaridade" in output


def test_top_k_option_limits_how_many_results_are_printed(workspace, capsys):
    run("index")
    capsys.readouterr()

    run("search", "container", "--top-k", "1")

    output = capsys.readouterr().out
    assert "1. " in output
    assert "2. " not in output


def test_search_without_an_index_reports_how_to_create_it(workspace, capsys):
    assert run("search", "container") == 1

    assert "python -m semantic_search index" in capsys.readouterr().err


def test_index_with_missing_documents_folder_reports_the_error(workspace, monkeypatch, capsys):
    monkeypatch.setenv("DOCUMENTS_DIR", str(workspace / "nao-existe"))

    assert run("index") == 1

    assert "não encontrada" in capsys.readouterr().err


def test_searching_with_another_model_than_the_indexed_one_is_refused(
    workspace, monkeypatch, capsys
):
    run("index")
    capsys.readouterr()
    monkeypatch.setenv("EMBEDDING_MODEL", "outro/modelo")

    assert run("search", "container") == 1

    assert "outro/modelo" in capsys.readouterr().err


@pytest.mark.parametrize(
    "bad_args",
    [["search", "   "], ["search", "x", "--top-k", "0"], ["search", "x", "--top-k", "abc"], []],
)
def test_invalid_usage_exits_with_argparse_error_code(workspace, bad_args):
    with pytest.raises(SystemExit) as exit_info:
        run(*bad_args)

    assert exit_info.value.code == 2
