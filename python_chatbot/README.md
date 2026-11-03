# Python Chatbot — tutor de Python com LangChain e OpenAI

Chatbot de terminal que responde perguntas sobre programação em Python usando um modelo de
linguagem da OpenAI. O LangChain monta o fluxo (prompt, histórico da conversa, modelo e
tratamento da saída).

## Arquitetura

```
chatbot/
├── config.py   # Settings e leitura das variáveis de ambiente / .env
├── chain.py    # prompt + chain LangChain, classe Chatbot (histórico) e tradução de erros da OpenAI
├── cli.py      # loop de terminal: lê a pergunta, chama o Chatbot, imprime a resposta
└── __main__.py # permite rodar com: python -m chatbot
tests/
├── fakes.py          # LLM falso que também grava o que o modelo recebeu
├── test_chatbot.py   # chain, histórico, erros e formato da requisição à OpenAI
├── test_config.py
└── test_cli.py
examples/sample_session.md   # perguntas sugeridas + espaço para a saída real
```

Fluxo de uma pergunta:

```
CLI --> Chatbot.ask --> ChatPromptTemplate(system + histórico + pergunta) --> ChatOpenAI --> StrOutputParser --> texto
```

O `Chatbot` recebe o LLM por parâmetro, então os testes trocam o `ChatOpenAI` por um falso sem
mexer em nada mais.

## Tecnologias

- Python 3.10 ou superior
- `langchain-core` e `langchain-openai` (LCEL: `prompt | modelo | parser`)
- `python-dotenv` (leitura do `.env`)
- LangSmith (opcional, só por variáveis de ambiente)
- Pytest

## Pré-requisitos

- Python 3.10+
- Uma chave de API da OpenAI, em https://platform.openai.com/api-keys. A conta precisa ter
  crédito: a API é cobrada por uso, separadamente de qualquer assinatura do ChatGPT.

## Instalação

PowerShell (Windows):

```powershell
cd python_chatbot
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação ("a execução de scripts foi desabilitada"), rode
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` e ative de novo. Isso vale só para
a janela atual.

Linux / macOS:

```bash
cd python_chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuração

Copie o modelo de configuração e preencha a chave:

```powershell
Copy-Item .env.example .env      # Linux/macOS: cp .env.example .env
notepad .env
```

No `.env`, preencha `OPENAI_API_KEY=` com a sua chave, sem aspas. O arquivo precisa se chamar
exatamente `.env`. Cuidado para o Bloco de Notas não salvar como `.env.txt`. **Nunca versione o
`.env`**: ele já está no `.gitignore`.

| Variável            | Obrigatória | Padrão        | Descrição                                           |
|---------------------|-------------|---------------|-----------------------------------------------------|
| `OPENAI_API_KEY`    | sim         | -             | Chave da API da OpenAI                              |
| `OPENAI_MODEL`      | não         | `gpt-6-luna`  | Modelo usado (veja "Decisões de projeto")           |
| `MAX_HISTORY_TURNS` | não         | `10`          | Quantos turnos (pergunta + resposta) o bot lembra   |
| `REQUEST_TIMEOUT`   | não         | `60`          | Tempo máximo, em segundos, de cada chamada          |

## Como executar

```bash
python -m chatbot
```

```
Tutor de Python (modelo: gpt-6-luna)
Digite sua pergunta, ou 'sair' para encerrar.

Você: Como criar uma lista em Python?
```

Para encerrar: `sair`, `exit`, `quit`, Ctrl+C ou Ctrl+D.

Perguntas sugeridas, incluindo uma de acompanhamento e uma fora do escopo, estão em
[`examples/sample_session.md`](examples/sample_session.md), com um espaço para colar a saída real.

## LangSmith (opcional)

Para ver o trace de cada pergunta (prompt enviado, resposta, latência e tokens), crie uma conta
em https://smith.langchain.com e acrescente ao `.env`:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=sua-chave-do-langsmith
LANGSMITH_PROJECT=python-chatbot
```

Não é preciso mudar o código. Sem essas variáveis, nada é enviado ao LangSmith. Cada pergunta
aparece como um trace chamado `python_tutor`.

## Como executar os testes

```bash
python -m pytest -v
```

Os testes não usam rede nem chave: trocam o modelo por um falso. Eles validam o fluxo do
programa, não a qualidade das respostas do modelo.

## Decisões de projeto

- **Modelo padrão `gpt-6-luna`, e não GPT-4.** O enunciado cita o GPT-4 como exemplo, mas a
  página de deprecações da OpenAI informa que GPT-4o, GPT-4o mini, GPT-4.1 e GPT-4.1 mini saem da
  API em 14/01/2027, e recomenda o `gpt-6-luna` como substituto. Escolher um modelo que já está
  de saída deixaria o projeto quebrado em pouco tempo. O modelo é só uma variável
  (`OPENAI_MODEL`), então trocar é imediato.
- **Histórico como lista simples, não `RunnableWithMessageHistory`.** Na versão do LangChain
  usada aqui (`langchain-core` 1.7.3), `RunnableWithMessageHistory` e
  `InMemoryChatMessageHistory` estão depreciados em favor da persistência do LangGraph. Para uma
  única conversa em memória, o LangGraph seria dependência e complexidade sem ganho. O
  `Chatbot` guarda as últimas mensagens e as entrega ao `MessagesPlaceholder` do prompt. Se o
  projeto crescer para várias sessões, persistência ou agentes, o caminho natural é o LangGraph.
- **O turno só entra no histórico depois do sucesso.** Uma chamada que falhou não deixa uma
  pergunta sem resposta na conversa (há teste para isso).
- **Histórico limitado (`MAX_HISTORY_TURNS`).** Só as últimas mensagens são guardadas e enviadas
  ao modelo, para controlar tokens, custo e memória.
- **`temperature` e `max_tokens` não são enviados.** Modelos de raciocínio da OpenAI costumam
  restringir esses parâmetros, e os padrões do provedor servem para um tutor. Há um teste que
  confere o corpo da requisição.
- **Erros da OpenAI viram mensagens acionáveis** (chave inválida, modelo inexistente, limite ou
  crédito esgotado, timeout, sem conexão). Só erros do provedor são traduzidos; qualquer outro
  é um bug e aparece normalmente.
- **Sem camadas extras:** uma classe (`Chatbot`) e uma função de montagem da chain bastam para
  este escopo.

## Limitações conhecidas

- **Não foi validado contra a API real da OpenAI.** O ambiente em que o projeto foi construído
  não tem acesso a ela. O que foi verificado: o corpo exato da requisição que o `ChatOpenAI`
  envia (com transporte HTTP simulado), o tratamento de cada tipo de erro e o fluxo do terminal.
  A primeira conversa real é a sua, por isso o arquivo de exemplos vem vazio.
- **O histórico some ao fechar o programa.** É memória da sessão, de propósito.
- **Restringir o assunto a Python é uma instrução de prompt**, não uma garantia técnica. O
  modelo pode, em casos raros, responder fora do escopo.
- **Custo:** cada pergunta é cobrada por tokens, e o histórico enviado a cada chamada faz as
  conversas longas ficarem mais caras. `MAX_HISTORY_TURNS` limita isso.
