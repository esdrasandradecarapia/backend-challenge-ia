import pytest

from chatbot import config
from chatbot.config import DEFAULT_MODEL, ConfigError, load_settings

ENV_VARS = ("OPENAI_API_KEY", "OPENAI_MODEL", "MAX_HISTORY_TURNS", "REQUEST_TIMEOUT")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Isola o teste do ambiente real e de qualquer .env que exista na pasta."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda: None)


def test_missing_api_key_raises_config_error():
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        load_settings()


def test_blank_api_key_counts_as_missing(monkeypatch):
    # É o que acontece quando o .env.example é copiado sem preencher a chave.
    monkeypatch.setenv("OPENAI_API_KEY", "   ")

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        load_settings()


def test_defaults_when_only_the_key_is_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")

    settings = load_settings()

    assert settings.openai_api_key == "sk-teste"
    assert settings.model == DEFAULT_MODEL
    assert settings.max_history_turns == 10
    assert settings.request_timeout == 60.0


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")
    monkeypatch.setenv("OPENAI_MODEL", "outro-modelo")
    monkeypatch.setenv("MAX_HISTORY_TURNS", "3")
    monkeypatch.setenv("REQUEST_TIMEOUT", "12.5")

    settings = load_settings()

    assert settings.model == "outro-modelo"
    assert settings.max_history_turns == 3
    assert settings.request_timeout == 12.5


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MAX_HISTORY_TURNS", "abc"),
        ("MAX_HISTORY_TURNS", "0"),
        ("MAX_HISTORY_TURNS", "-2"),
        ("MAX_HISTORY_TURNS", "2.5"),
        ("REQUEST_TIMEOUT", "rápido"),
        ("REQUEST_TIMEOUT", "0"),
    ],
)
def test_invalid_numeric_values_raise_config_error(monkeypatch, name, value):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigError, match=name):
        load_settings()
