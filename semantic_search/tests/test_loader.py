import pytest

from semantic_search.errors import DocumentLoadError
from semantic_search.loader import load_documents


def write(directory, name, content, encoding="utf-8"):
    (directory / name).write_text(content, encoding=encoding)


def test_loads_title_and_text_from_each_txt_file(tmp_path):
    write(tmp_path, "docker.txt", "Containers\n\nUm container empacota a aplicação.\n")

    [document] = load_documents(tmp_path)

    assert document.id == "docker"
    assert document.title == "Containers"
    assert document.text == "Um container empacota a aplicação."
    assert document.source == "docker.txt"


def test_documents_come_sorted_by_file_name_and_other_files_are_ignored(tmp_path):
    write(tmp_path, "b.txt", "Segundo\ntexto b")
    write(tmp_path, "a.txt", "Primeiro\ntexto a")
    write(tmp_path, "leia-me.md", "não é documento")

    documents = load_documents(tmp_path)

    assert [d.id for d in documents] == ["a", "b"]


def test_accepts_files_saved_with_bom_and_windows_line_endings(tmp_path):
    (tmp_path / "doc.txt").write_bytes("\ufeffTítulo\r\n\r\nTexto com acentuação.".encode("utf-8"))

    [document] = load_documents(tmp_path)

    assert document.title == "Título"
    assert document.text == "Texto com acentuação."


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(DocumentLoadError, match="não encontrada"):
        load_documents(tmp_path / "nao-existe")


def test_directory_without_txt_files_is_reported(tmp_path):
    with pytest.raises(DocumentLoadError, match="Nenhum arquivo .txt"):
        load_documents(tmp_path)


@pytest.mark.parametrize("content", ["", "   \n  ", "Só o título"])
def test_file_without_title_and_text_is_rejected_naming_the_file(tmp_path, content):
    write(tmp_path, "incompleto.txt", content)

    with pytest.raises(DocumentLoadError, match="incompleto.txt"):
        load_documents(tmp_path)


def test_file_that_is_not_utf8_is_rejected_naming_the_file(tmp_path):
    (tmp_path / "latin.txt").write_bytes("Título\ncoração".encode("latin-1"))

    with pytest.raises(DocumentLoadError, match="latin.txt"):
        load_documents(tmp_path)
