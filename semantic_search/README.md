# Semantic Search — busca semântica com embeddings e FAISS

Sistema de busca semântica em documentos de texto. Dada uma consulta em linguagem natural, ele
devolve os documentos de significado mais próximo, mesmo quando a consulta não usa as mesmas
palavras do texto.

```
python -m semantic_search search "fazer meu programa funcionar igual em qualquer computador"
```

## Como o processo funciona

```
 INDEXAÇÃO (comando index)
 .txt --> loader --> Document --> embedder --> vetores --> FAISS --> index/
                                  (modelo)    (n x d)    (vectors.faiss + metadata.json)

 BUSCA (comando search)
 consulta --> embedder --> vetor (d) --> FAISS (similaridade) --> posições --> Documents + score
```

### 1. Documentos

O corpus fica em `data/documents/`: 15 textos curtos, originais e em português, sobre temas de
tecnologia (containers, bancos de dados, segurança, machine learning etc.). Cada arquivo `.txt`
tem o **título na primeira linha** e o texto depois dela. Para usar outro conjunto, basta trocar
os arquivos ou apontar `DOCUMENTS_DIR` para outra pasta.

### 2. Geração dos embeddings

Um embedding é um vetor de números que representa o significado de um texto: textos de sentido
parecido ficam próximos no espaço, mesmo com palavras diferentes. Para cada lote de textos, o
`TransformersEmbedder` faz:

1. **Tokenização:** o tokenizador converte o texto em ids de tokens, truncando no limite do
   modelo e completando com padding os textos mais curtos do lote.
2. **Modelo:** o transformer produz um vetor para cada token.
3. **Mean pooling:** tira a média dos vetores dos tokens **reais**, usando a máscara de atenção
   para ignorar o padding. O resultado é um único vetor por texto.
4. **Normalização L2:** deixa o vetor com norma 1.

O modelo padrão é o [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small),
multilíngue e pequeno o bastante para rodar na CPU. Ele é baixado do Hugging Face na primeira
execução (cerca de 470 MB) e fica em cache. Os modelos da família E5 foram treinados com
prefixos: `query: ` antes da consulta e `passage: ` antes do documento. O projeto os aplica
sozinho; para um modelo que não use prefixos, defina as variáveis `EMBEDDING_QUERY_PREFIX` e
`EMBEDDING_DOCUMENT_PREFIX` como vazias.

O texto indexado de cada documento é o **título + o corpo**, porque o título resume o assunto.

### 3. Armazenamento na vector store

Os vetores vão para um índice FAISS `IndexFlatIP`:

- **Produto interno + vetores normalizados = similaridade de cosseno.** Por isso a normalização
  do passo 4 é parte do contrato do embedder, e o `FaissVectorStore` recusa vetores que não
  tenham norma 1.
- **Busca exata:** a consulta é comparada com todos os vetores. Para 15 documentos (ou
  milhares) é o mais simples e não perde resultado. Com milhões de vetores, valeria um índice
  aproximado (HNSW, IVF).

### 4. Persistência

O comando `index` grava, na pasta `index/`:

| Arquivo         | Conteúdo |
|-----------------|----------|
| `vectors.faiss` | Os vetores. |
| `metadata.json` | Os documentos, na **mesma ordem** dos vetores, e o nome do modelo usado. |

O vetor na posição *i* representa o documento *i*. Ao carregar, o programa confere se o modelo
atual é o mesmo do índice (vetores de modelos diferentes não são comparáveis) e se vetores e
documentos têm o mesmo tamanho. Em caso de divergência, ele para com uma mensagem pedindo para
reindexar.

### 5. Busca

`SemanticSearchEngine.search(consulta, top_k)` gera o embedding da consulta, pede ao FAISS os
`top_k` vetores mais similares e devolve os documentos correspondentes, do mais para o menos
relevante, cada um com o seu score (similaridade de cosseno).

```python
results = engine.search("deixar as consultas ao banco mais rápidas", top_k=3)
for result in results:
    print(result.document.title, result.score)
```

## Arquitetura

```
semantic_search/
├── models.py                # Document e SearchResult
├── loader.py                # lê os .txt
├── embedder.py              # contrato Embedder (Protocol)
├── transformers_embedder.py # implementação com transformers + PyTorch
├── vector_store.py          # FaissVectorStore: add, search, save, load
├── index_repository.py      # grava e lê o índice no disco
├── engine.py                # SemanticSearchEngine: orquestra indexação e busca
├── config.py                # configuração por variáveis de ambiente
├── errors.py                # erros esperados do domínio
└── cli.py                   # comandos index e search
data/documents/              # o corpus
tests/
examples/demo_queries.md     # consultas de demonstração e espaço para a saída real
```

### SOLID e código limpo, de forma pragmática

| Princípio | Como aparece no projeto |
|-----------|-------------------------|
| **S**ingle Responsibility | Cada módulo faz uma coisa: o `loader` lê, o `embedder` vetoriza, o `vector_store` busca vetores, o `index_repository` persiste, o `engine` orquestra e a `cli` conversa com o usuário. |
| **O**pen/Closed | Para usar outro gerador de embeddings (uma API, outro modelo), basta uma nova classe que cumpra o `Embedder`. O `engine` não muda. |
| **L**iskov Substitution | Qualquer `Embedder` substitui outro. Os testes trocam o modelo real por um falso sem tocar no `engine`. |
| **I**nterface Segregation | O `Embedder` tem só dois métodos, `embed_documents` e `embed_query`, exatamente o que o `engine` usa. |
| **D**ependency Inversion | O `engine` depende da abstração `Embedder`. A implementação concreta é escolhida na `cli` (`embedder_factory`), na borda do sistema. |

Outras práticas: tipagem em todo o código, erros de domínio próprios com mensagens que dizem o
que fazer, validação nas fronteiras (consulta vazia, `top-k` inválido, vetores sem
normalização, índice incompatível), configuração fora do código e comentários só para explicar o
porquê das decisões.

## Tecnologias

- Python 3.10+
- `transformers` + PyTorch: modelo de embeddings
- `faiss-cpu`: vector store
- `numpy`
- `pytest` e `ruff`: testes e qualidade de código

## Instalação

PowerShell (Windows):

```powershell
cd semantic_search
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS:

```bash
cd semantic_search
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A instalação do PyTorch é a parte demorada. No Linux, o pacote padrão do PyPI inclui bibliotecas
CUDA (vários GB). Se você não usa GPU, instale antes a versão só de CPU:
`pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`.

## Como executar

```bash
python -m semantic_search index
python -m semantic_search search "fazer meu programa funcionar igual em qualquer computador"
python -m semantic_search search "deixar as consultas ao banco mais rápidas" --top-k 5
```

O primeiro comando baixa o modelo (só na primeira vez), gera os embeddings e cria o índice em
`index/`. Os seguintes só consultam. Cada `search` carrega o modelo de novo, o que leva alguns
segundos.

Exemplo do formato da saída:

```
Consulta: <texto da consulta>

1. <título do documento>  (similaridade 0.000)
   <arquivo>.txt
   <início do texto>...
```

Consultas de demonstração, com o resultado esperado de cada uma e espaço para a saída real, em
[`examples/demo_queries.md`](examples/demo_queries.md).

## Configuração

Tudo é opcional. Variáveis de ambiente:

| Variável                    | Padrão                          | Descrição |
|-----------------------------|---------------------------------|-----------|
| `DOCUMENTS_DIR`             | `data/documents`                | Pasta com os `.txt` |
| `INDEX_DIR`                 | `index`                         | Pasta onde o índice é gravado |
| `EMBEDDING_MODEL`           | `intfloat/multilingual-e5-small`| Modelo do Hugging Face |
| `EMBEDDING_QUERY_PREFIX`    | `query: `                       | Prefixo das consultas (E5) |
| `EMBEDDING_DOCUMENT_PREFIX` | `passage: `                     | Prefixo dos documentos (E5) |

No PowerShell: `$env:EMBEDDING_MODEL = "outro/modelo"`. Trocar o modelo exige rodar `index`
de novo. O programa detecta a diferença e avisa.

## Problemas comuns

- **Aviso sobre `symlinks` ao baixar o modelo (Windows):** é inofensivo. O cache do Hugging Face
  só usa um pouco mais de disco quando o Windows não está em modo desenvolvedor.
- **Erro de conexão no primeiro `index`:** o download do modelo precisa de internet. Depois de
  baixado, fica em cache e o uso é offline.
- **`Índice não encontrado`:** rode `python -m semantic_search index` antes de `search`.

## Testes

```bash
python -m pytest -v                  # testes rápidos, sem internet nem download
python -m pytest -m integration -v   # com o modelo real (baixa ~470 MB)
ruff check .                         # qualidade de código
ruff format --check .                # formatação
```

Os testes padrão cobrem:

- **FAISS de verdade:** ordenação, `top_k`, índice vazio, dimensão errada, vetores não
  normalizados, gravação e leitura (inclusive em pasta com acento).
- **Persistência:** ida e volta, índice ausente, JSON inválido, campos faltando, arquivo de
  vetores corrompido, modelo diferente e vetores/documentos fora de sincronia.
- **Motor de busca:** ranking, scores em ordem, `top_k`, consulta vazia e o título entrando no
  texto indexado, usando um embedder falso e determinístico.
- **`TransformersEmbedder`, sem download:** um modelo BERT minúsculo com pesos aleatórios é
  montado em pasta temporária e carregado pelo mesmo caminho do modelo real. Os testes conferem
  formato, tipo, norma 1, independência do tamanho do lote (o padding é ignorado), prefixos,
  truncamento e o mean pooling.
- **CLI**, nos fluxos de sucesso e de erro.

O teste de integração confere, com o modelo real, que cada consulta de
[`examples/demo_queries.md`](examples/demo_queries.md) traz o documento esperado entre os 3
primeiros.

## Limitações conhecidas

- **Não foi validado com o modelo real de embeddings.** O ambiente em que o projeto foi
  construído não acessa o Hugging Face. Tudo que não depende dos pesos do modelo foi testado,
  mas a qualidade semântica das respostas só existe com o modelo treinado. Por isso o teste de
  integração e o arquivo de demonstração são o seu passo de conferência, e o segundo vem vazio.
- **A busca sempre devolve `top-k` resultados**, mesmo que nenhum seja relevante. Não há limiar
  de corte porque os scores do E5 costumam ficar concentrados numa faixa alta, e um valor de
  corte fixo seria arbitrário sem medir o seu corpus. O que vale é a ordem relativa. Se for
  preciso filtrar irrelevantes, o caminho é calibrar um limiar com consultas reais.
- **Busca exata em memória:** adequada a corpora pequenos e médios. Para milhões de documentos
  seria preciso um índice aproximado.
- **O índice guarda o texto dos documentos** no `metadata.json`, o que o torna independente da
  pasta original, mas duplica o conteúdo.
- **Sem abstração para o vector store.** O `engine` usa o `FaissVectorStore` diretamente. Com uma
  única implementação, uma interface seria especulativa. Se surgir um segundo backend (Milvus,
  por exemplo), o momento de extraí-la é esse.
