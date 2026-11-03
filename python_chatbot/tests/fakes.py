"""LLM falso para os testes: sem rede, sem chave, sem custo."""

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda


def make_fake_llm(*replies: str | Exception):
    """Devolve (llm, calls).

    O llm responde na ordem dada (ou levanta, se o item for uma exceção).
    `calls` guarda as mensagens que o modelo recebeu em cada chamada, o que
    permite verificar system prompt, histórico e corte do histórico.
    """
    calls: list[list[BaseMessage]] = []
    pending = iter(replies)

    def fake(prompt_value):
        calls.append(prompt_value.to_messages())
        reply = next(pending)
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=reply)

    return RunnableLambda(fake), calls
