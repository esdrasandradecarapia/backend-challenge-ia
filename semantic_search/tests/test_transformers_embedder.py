"""Testa o TransformersEmbedder de verdade, mas sem baixar nada.

Um modelo BERT minúsculo, com pesos aleatórios, é salvo em uma pasta temporária e
carregado pelo mesmo caminho de código usado com o modelo real. Isso valida carga,
tokenização, pooling, normalização e prefixos. A qualidade semântica não é testada
aqui; ela só existe em modelos treinados (veja test_real_model.py).
"""

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import AutoTokenizer, BertConfig, BertModel, PreTrainedTokenizerFast

from semantic_search.transformers_embedder import TransformersEmbedder, mean_pool

HIDDEN_SIZE = 16
VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "query", "passage", ":", "docker", "container", "banco", "dados", "consulta",
]  # fmt: skip
TEXTS = ["docker container", "banco", "docker container banco dados consulta"]


def _build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Tokenizador WordPiece com o vocabulário de VOCAB.

    Montado com a biblioteca `tokenizers` porque BertTokenizerFast(vocab_file=...) não
    respeita o vocabulário informado em todas as versões do transformers.
    """
    ids = {token: index for index, token in enumerate(VOCAB)}
    backend = Tokenizer(models.WordPiece(ids, unk_token="[UNK]"))
    backend.normalizer = normalizers.BertNormalizer(lowercase=True)
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", ids["[CLS]"]), ("[SEP]", ids["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny-bert")
    _build_tiny_tokenizer().save_pretrained(directory)

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    BertModel(config).save_pretrained(directory)
    return str(directory)


def test_tiny_tokenizer_really_knows_the_test_vocabulary(tiny_model_dir):
    """Guarda do próprio fixture: palavras do vocabulário não podem virar [UNK]."""
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)

    ids = tokenizer("query: docker banco")["input_ids"]

    assert tokenizer.unk_token_id not in ids
    assert tokenizer.convert_ids_to_tokens(ids) == [
        "[CLS]",
        "query",
        ":",
        "docker",
        "banco",
        "[SEP]",
    ]


@pytest.fixture(scope="module")
def embedder(tiny_model_dir):
    return TransformersEmbedder(tiny_model_dir, max_length=32)


def test_documents_become_float32_unit_vectors_of_the_model_dimension(embedder):
    vectors = embedder.embed_documents(TEXTS)

    assert vectors.shape == (3, HIDDEN_SIZE)
    assert vectors.dtype == np.float32
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_query_is_a_single_unit_vector(embedder):
    vector = embedder.embed_query("docker container")

    assert vector.shape == (HIDDEN_SIZE,)
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_result_does_not_depend_on_batch_size_so_padding_is_ignored(tiny_model_dir):
    one_by_one = TransformersEmbedder(tiny_model_dir, batch_size=1, max_length=32)
    all_together = TransformersEmbedder(tiny_model_dir, batch_size=8, max_length=32)

    np.testing.assert_allclose(
        one_by_one.embed_documents(TEXTS), all_together.embed_documents(TEXTS), atol=1e-5
    )


def test_same_text_always_gives_the_same_vector(embedder):
    np.testing.assert_array_equal(embedder.embed_query("banco"), embedder.embed_query("banco"))


def test_different_texts_give_different_vectors(embedder):
    vectors = embedder.embed_documents(TEXTS)

    assert not np.allclose(vectors[0], vectors[1])


def test_prefixes_are_applied_to_queries_and_documents(tiny_model_dir):
    plain = TransformersEmbedder(tiny_model_dir, max_length=32)
    prefixed = TransformersEmbedder(
        tiny_model_dir, query_prefix="query: ", document_prefix="passage: ", max_length=32
    )

    assert not np.allclose(plain.embed_query("banco"), prefixed.embed_query("banco"))
    assert not np.allclose(plain.embed_documents(["banco"]), prefixed.embed_documents(["banco"]))
    assert not np.allclose(prefixed.embed_query("banco"), prefixed.embed_documents(["banco"])[0])


def test_without_prefixes_query_and_document_are_embedded_the_same_way(embedder):
    np.testing.assert_allclose(
        embedder.embed_query("banco"), embedder.embed_documents(["banco"])[0], atol=1e-6
    )


def test_no_texts_gives_an_empty_array_with_the_right_dimension(embedder):
    assert embedder.embed_documents([]).shape == (0, HIDDEN_SIZE)


def test_text_longer_than_max_length_is_truncated_instead_of_rejected(tiny_model_dir):
    short_limit = TransformersEmbedder(tiny_model_dir, max_length=8)

    vector = short_limit.embed_query("docker " * 200)

    assert vector.shape == (HIDDEN_SIZE,)
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_invalid_batch_size_is_rejected(tiny_model_dir):
    with pytest.raises(ValueError, match="batch_size"):
        TransformersEmbedder(tiny_model_dir, batch_size=0)


def test_mean_pool_averages_only_real_tokens():
    token_vectors = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    attention_mask = torch.tensor([[1, 1, 0]])  # o terceiro token é padding

    pooled = mean_pool(token_vectors, attention_mask)

    assert pooled.tolist() == [[2.0, 2.0]]
