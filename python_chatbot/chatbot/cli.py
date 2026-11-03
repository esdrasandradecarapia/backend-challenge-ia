"""Interface de linha de comando: lê perguntas do terminal e imprime as respostas."""

import sys

from chatbot.chain import Chatbot, ChatbotError, create_llm
from chatbot.config import ConfigError, load_settings

EXIT_WORDS = {"sair", "exit", "quit"}


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Erro de configuração: {exc}", file=sys.stderr)
        return 2

    bot = Chatbot(create_llm(settings), max_history_turns=settings.max_history_turns)

    print(f"Tutor de Python (modelo: {settings.model})")
    print("Digite sua pergunta, ou 'sair' para encerrar.\n")

    while True:
        try:
            question = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C
            print()
            break

        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            break

        print("\nPensando...", flush=True)
        try:
            answer = bot.ask(question)
        except KeyboardInterrupt:
            print("\nInterrompido.")
            break
        except ChatbotError as exc:
            print(f"\n[erro] {exc}\n", file=sys.stderr)
            continue

        print(f"\nTutor:\n{answer}\n")

    return 0
