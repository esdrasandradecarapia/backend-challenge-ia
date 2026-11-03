import httpx
import openai

from chatbot import cli
from chatbot.config import ConfigError, Settings
from tests.fakes import make_fake_llm


def run_cli(monkeypatch, inputs, llm):
    """Executa cli.main() com o teclado simulado e o LLM falso."""
    monkeypatch.setattr(cli, "load_settings", lambda: Settings(openai_api_key="sk-teste"))
    monkeypatch.setattr(cli, "create_llm", lambda settings: llm)
    typed = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(typed))
    return cli.main()


def test_prints_the_answer_and_ignores_blank_lines(monkeypatch, capsys):
    llm, calls = make_fake_llm("Use colchetes.")

    code = run_cli(monkeypatch, ["", "Como criar uma lista?", "sair"], llm)

    assert code == 0
    assert "Use colchetes." in capsys.readouterr().out
    assert len(calls) == 1  # a linha em branco não chamou o modelo


def test_error_is_reported_and_the_conversation_continues(monkeypatch, capsys):
    request = httpx.Request("POST", "https://api.openai.com")
    llm, _ = make_fake_llm(
        openai.APIConnectionError(request=request),
        "Segunda resposta.",
    )

    code = run_cli(monkeypatch, ["pergunta 1", "pergunta 2", "sair"], llm)

    captured = capsys.readouterr()
    assert code == 0
    assert "conectar" in captured.err
    assert "Segunda resposta." in captured.out


def test_end_of_input_exits_cleanly(monkeypatch):
    llm, _ = make_fake_llm()

    def raise_eof(prompt=""):
        raise EOFError

    monkeypatch.setattr(cli, "load_settings", lambda: Settings(openai_api_key="sk-teste"))
    monkeypatch.setattr(cli, "create_llm", lambda settings: llm)
    monkeypatch.setattr("builtins.input", raise_eof)

    assert cli.main() == 0


def test_missing_configuration_exits_with_code_2(monkeypatch, capsys):
    def fail():
        raise ConfigError("OPENAI_API_KEY não definida.")

    monkeypatch.setattr(cli, "load_settings", fail)

    assert cli.main() == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err
