# Consultas de demonstração

Este arquivo guarda uma execução **real** da busca semântica. Os testes automatizados não
usam o modelo de verdade, então os resultados só existem depois que você roda o projeto na
sua máquina. Nada aqui foi escrito à mão ou inventado.

## O que observar

As consultas abaixo evitam de propósito as palavras-chave dos textos. Em uma busca por
palavras, nenhuma delas acharia o documento certo. Na busca semântica, ele deve aparecer
entre os primeiros.

| # | Consulta | Documento esperado |
|---|----------|--------------------|
| 1 | `fazer meu programa funcionar igual em qualquer computador` | `containers-docker` |
| 2 | `deixar as consultas ao banco de dados mais rápidas` | `indices-banco-dados` (e provavelmente `cache-redis`) |
| 3 | `impedir que um atacante execute comandos maliciosos pelo formulário do site` | `seguranca-sql-injection` |
| 4 | `descobrir por que o sistema está lento em produção` | `observabilidade` |
| 5 | `ensinar o computador a reconhecer padrões a partir de exemplos` | `aprendizado-supervisionado` |
| 6 | `garantir que cada mudança no código seja verificada antes de ir ao ar` | `ci-cd` (e provavelmente `testes-automatizados`) |
| 7 | `um assistente que responde perguntas usando os documentos da minha empresa` | `llms-rag` |
| 8 | `receita de bolo de cenoura` | nenhum: consulta fora do assunto |

A consulta 8 mostra o comportamento sem relevância: a busca sempre devolve os `top-k` mais
próximos, mesmo quando nenhum faz sentido. Veja a seção "Limitações" do README.

## Como gerar

```powershell
python -m semantic_search index
python -m semantic_search search "fazer meu programa funcionar igual em qualquer computador"
python -m semantic_search search "deixar as consultas ao banco de dados mais rápidas"
python -m semantic_search search "impedir que um atacante execute comandos maliciosos pelo formulário do site"
python -m semantic_search search "descobrir por que o sistema está lento em produção"
python -m semantic_search search "ensinar o computador a reconhecer padrões a partir de exemplos"
python -m semantic_search search "garantir que cada mudança no código seja verificada antes de ir ao ar"
python -m semantic_search search "um assistente que responde perguntas usando os documentos da minha empresa"
python -m semantic_search search "receita de bolo de cenoura"
```

Para conferir os resultados esperados de forma automática (baixa o modelo na primeira vez):

```powershell
python -m pytest -m integration -v
```

## Saída real

<!-- Cole aqui a saída do terminal. -->

```text

```
