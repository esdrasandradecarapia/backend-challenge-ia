import json

import httpx
import openai
import pytest
from langchain_openai import ChatOpenAI

from chatbot.chain import SYSTEM_PROMPT, Chatbot, ChatbotError, create_llm
from chatbot.config import Settings
from tests.fakes import make_fake_llm


def contents(messages):
    return [m.content for m in messages]


def status_error(cls, status: int):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls("erro simulado", response=response, body=None)


# ---------- Fluxo básico ----------


def test_ask_returns_the_model_answer_as_text():
    llm, _ = make_fake_llm("Use colchetes: minha_lista = [1, 2, 3]")
    bot = Chatbot(llm)

    assert bot.ask("Como criar uma lista em Python?") == "Use colchetes: minha_lista = [1, 2, 3]"


def test_model_receives_system_prompt_then_the_question():
    llm, calls = make_fake_llm("ok")

    Chatbot(llm).ask("  Como criar uma lista em Python?  ")

    assert contents(calls[0]) == [SYSTEM_PROMPT, "Como criar uma lista em Python?"]


def test_blank_question_is_rejected_without_calling_the_model():
    llm, calls = make_fake_llm("não deve ser usada")

    with pytest.raises(ChatbotError):
        Chatbot(llm).ask("   ")

    assert calls == []


# ---------- Histórico da conversa ----------


def test_previous_turns_are_sent_on_the_next_question():
    llm, calls = make_fake_llm("resposta 1", "resposta 2")
    bot = Chatbot(llm)

    bot.ask("pergunta 1")
    bot.ask("pergunta 2")

    assert contents(calls[1]) == [SYSTEM_PROMPT, "pergunta 1", "resposta 1", "pergunta 2"]


def test_history_sent_to_the_model_is_limited_to_the_last_turns():
    llm, calls = make_fake_llm("r1", "r2", "r3")
    bot = Chatbot(llm, max_history_turns=1)

    bot.ask("p1")
    bot.ask("p2")
    bot.ask("p3")

    # Só o turno anterior (p2/r2) entra; p1/r1 ficou de fora.
    assert contents(calls[2]) == [SYSTEM_PROMPT, "p2", "r2", "p3"]


def test_invalid_history_limit_is_rejected():
    llm, _ = make_fake_llm()

    with pytest.raises(ValueError):
        Chatbot(llm, max_history_turns=0)


def test_failed_call_does_not_pollute_the_history():
    llm, calls = make_fake_llm(
        status_error(openai.RateLimitError, 429),
        "resposta 2",
    )
    bot = Chatbot(llm)

    with pytest.raises(ChatbotError):
        bot.ask("pergunta que falhou")
    answer = bot.ask("pergunta 2")

    assert answer == "resposta 2"
    assert contents(calls[1]) == [SYSTEM_PROMPT, "pergunta 2"]


# ---------- Tradução de erros da OpenAI ----------


@pytest.mark.parametrize(
    ("error", "expected_hint"),
    [
        (status_error(openai.AuthenticationError, 401), "OPENAI_API_KEY"),
        (status_error(openai.PermissionDeniedError, 403), "OPENAI_MODEL"),
        (status_error(openai.NotFoundError, 404), "OPENAI_MODEL"),
        (status_error(openai.RateLimitError, 429), "crédito"),
        (status_error(openai.InternalServerError, 500), "HTTP 500"),
        (status_error(openai.BadRequestError, 400), "HTTP 400"),
        (
            openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com")),
            "demorou",
        ),
        (
            openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")),
            "conectar",
        ),
    ],
    ids=["401", "403", "404", "429", "500", "400", "timeout", "conexao"],
)
def test_openai_errors_become_actionable_messages(error, expected_hint):
    llm, _ = make_fake_llm(error)

    with pytest.raises(ChatbotError, match=expected_hint):
        Chatbot(llm).ask("qualquer pergunta")


def test_unexpected_errors_are_not_swallowed():
    llm, _ = make_fake_llm(RuntimeError("bug"))

    with pytest.raises(RuntimeError, match="bug"):
        Chatbot(llm).ask("qualquer pergunta")


# ---------- Criação do LLM real (sem chamar a rede) ----------


def test_create_llm_uses_settings_and_leaves_sampling_params_unset():
    settings = Settings(openai_api_key="sk-teste", model="meu-modelo", request_timeout=30.0)

    llm = create_llm(settings)

    assert llm.model_name == "meu-modelo"
    assert llm.request_timeout == 30.0
    assert llm.temperature is None
    assert llm.max_tokens is None


def test_request_sent_to_openai_has_the_expected_shape():
    """Usa o ChatOpenAI real, mas com um transporte HTTP falso (sem rede)."""
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append((str(request.url), json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "meu-modelo",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"resposta {len(sent)}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    llm = ChatOpenAI(
        model="meu-modelo",
        api_key="sk-teste",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    bot = Chatbot(llm)

    assert bot.ask("Como criar uma lista em Python?") == "resposta 1"
    assert bot.ask("E como adiciono um item?") == "resposta 2"

    url, body = sent[1]
    assert url.endswith("/v1/chat/completions")
    assert body["model"] == "meu-modelo"
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "user"]
    # Parâmetros que modelos de raciocínio costumam restringir não devem ser enviados.
    assert not {"temperature", "max_tokens", "max_completion_tokens"} & body.keys()
