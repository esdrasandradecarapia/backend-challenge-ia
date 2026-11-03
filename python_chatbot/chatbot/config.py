"""Configuração lida de variáveis de ambiente (e do arquivo .env, se existir)."""

import os
from collections.abc import Callable
from dataclasses import dataclass

from dotenv import load_dotenv

# O enunciado cita o GPT-4, mas a OpenAI está desativando a família GPT-4 e
# recomenda o gpt-6-luna como substituto (veja o README). O padrão pode ser
# trocado sem mexer no código, via OPENAI_MODEL.
DEFAULT_MODEL = "gpt-6-luna"


class ConfigError(Exception):
    """Configuração ausente ou inválida, com mensagem pronta para o usuário."""


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    model: str = DEFAULT_MODEL
    max_history_turns: int = 10
    request_timeout: float = 60.0


def _positive_number(name: str, default: float, cast: Callable[[str], float]) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        raise ConfigError(f"{name} deve ser um número válido (recebido: {raw!r}).") from None
    if value <= 0:
        raise ConfigError(f"{name} deve ser maior que zero (recebido: {raw!r}).")
    return value


def load_settings() -> Settings:
    # load_dotenv exporta o .env para os.environ. Isso também é o que permite ao
    # LangChain enxergar as variáveis LANGSMITH_* e ligar o rastreamento.
    # Variáveis já definidas no sistema têm prioridade sobre o .env.
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            "OPENAI_API_KEY não definida. Copie .env.example para .env e preencha a chave."
        )

    return Settings(
        openai_api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_MODEL,
        max_history_turns=int(_positive_number("MAX_HISTORY_TURNS", 10, int)),
        request_timeout=float(_positive_number("REQUEST_TIMEOUT", 60.0, float)),
    )
