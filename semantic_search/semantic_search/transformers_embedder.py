"""Embeddings com a biblioteca `transformers` (modelo do Hugging Face + PyTorch).

Passo a passo para cada lote de textos:
  1. o tokenizador converte o texto em ids de tokens (com truncamento e padding);
  2. o modelo produz um vetor por token (last_hidden_state);
  3. o mean pooling tira a média dos vetores dos tokens reais (ignorando o padding),
     resultando em um único vetor por texto;
  4. a normalização L2 deixa o vetor com norma 1.
"""

from collections.abc import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class TransformersEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size deve ser pelo menos 1")
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._batch_size = batch_size
        self._max_length = max_length

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        # float32 explícito: dá o mesmo resultado em qualquer máquina, sem depender
        # do tipo em que o checkpoint foi salvo. eval() desliga o dropout.
        self._model = AutoModel.from_pretrained(model_name, dtype=torch.float32).eval()

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed([self._document_prefix + text for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self._query_prefix + text])[0]

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._model.config.hidden_size), dtype=np.float32)
        batches = [
            self._embed_batch(texts[start : start + self._batch_size])
            for start in range(0, len(texts), self._batch_size)
        ]
        return np.concatenate(batches)

    @torch.inference_mode()
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        token_vectors = self._model(**encoded).last_hidden_state
        pooled = mean_pool(token_vectors, encoded["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.numpy().astype(np.float32)


def mean_pool(token_vectors: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Média dos vetores dos tokens, desconsiderando o padding (máscara 0)."""
    mask = attention_mask.unsqueeze(-1).to(token_vectors.dtype)
    summed = (token_vectors * mask).sum(dim=1)
    token_count = mask.sum(dim=1).clamp(min=1e-9)
    return summed / token_count
