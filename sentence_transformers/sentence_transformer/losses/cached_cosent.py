from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import tqdm
from torch import Tensor, nn

from sentence_transformers.base.losses.gradcache import (
    CachedLossMixin,
    _validate_mini_batch_num_tokens,
    has_static_embedding_input,
)
from sentence_transformers.sentence_transformer.model import SentenceTransformer
from sentence_transformers.util import pairwise_cos_sim, similarity_fct_name


class CachedCoSENTLoss(CachedLossMixin, nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct: Callable[[Tensor, Tensor], Tensor] = pairwise_cos_sim,
        mini_batch_size: int = 32,
        mini_batch_num_tokens: int | None = None,
        show_progress_bar: bool = False,
    ) -> None:
        """
        Boosted version of :class:`CoSENTLoss` by GradCache (https://huggingface.co/papers/2101.06983).

        CoSENT ranks every input pair against every other input pair in the batch: the loss is a single
        logsumexp over up to ``batch_size * (batch_size - 1) / 2`` comparisons. A batch of 64 pairs gives
        about 2k comparisons, while a batch of 4096 pairs gives about 8.4M. Because every comparison sits
        inside the same sum, the batch cannot be split into independent mini-batches, and gradient
        accumulation would only compare the pairs within each mini-batch. GradCache solves this by
        dividing the computation into an embedding stage and a loss stage, which both run in mini-batches:

            (1) It first does a quick embedding step without gradients/computation graphs to get all the embeddings.
            (2) Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings.
            (3) A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain.

        The loss only depends on the embeddings through the ``batch_size`` pair scores, so step (2) is cheap.
        It processes the ``(batch_size, batch_size)`` comparison matrix ``mini_batch_size`` rows at a time,
        so that matrix is never materialized in full, and the result is exactly the loss and gradient of
        :class:`CoSENTLoss` on the whole batch.

        See :class:`CoSENTLoss` for more details about the underlying loss itself.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale value. Represents the inverse temperature.
            similarity_fct: Function to compute the PAIRWISE similarity between embeddings. Default is
                ``util.pairwise_cos_sim``. Use ``util.pairwise_angle_sim`` for a cached :class:`AnglELoss`
                on (input_A, input_B) pairs.
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the faster the training is, but the more memory
                is used. It's recommended to set it as high as your GPU memory allows. The default value is 32.
            mini_batch_num_tokens: If set, the embedding mini-batches are packed by total (non-padding)
                token count instead of by ``mini_batch_size`` sequences, which speeds up training on
                variable-length data. Most effective for models that avoid padded compute, e.g. flash
                attention with input flattening. See the Speeding up Inference documentation for details.
            show_progress_bar: If True, a progress bar for the mini-batches is shown during training. The default is False.

        References:
            - For further details, see: https://penghao-bdsc.github.io/papers/CoSENT_TASLP2024.pdf
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://huggingface.co/papers/2101.06983

        Requirements:
            1. Input pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].
            2. Should be used with large ``per_device_train_batch_size`` and low ``mini_batch_size`` for superior
               performance, but slower training time than :class:`CoSENTLoss`.

        Inputs:
            +--------------------------------+------------------------+
            | Inputs                         | Labels                 |
            +================================+========================+
            | (input_A, input_B) pairs       | float similarity score |
            +--------------------------------+------------------------+

        Relations:
            - Equivalent to :class:`CoSENTLoss`, but with caching that allows for much higher batch sizes
              without extra memory usage. This also lets it share a large ``per_device_train_batch_size`` with
              other cached losses like :class:`CachedMultipleNegativesRankingLoss` in multi-dataset training.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "sentence1": ["It's nice weather outside today.", "He drove to work."],
                    "sentence2": ["It's so sunny.", "She walked to the store."],
                    "score": [1.0, 0.3],
                })
                loss = losses.CachedCoSENTLoss(model, mini_batch_size=64)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        if has_static_embedding_input(model):
            raise ValueError(
                "CachedCoSENTLoss is not compatible with a SentenceTransformer model based on a "
                "StaticEmbedding, whose inputs cannot be split into mini-batches along a batch dimension. "
                "Consider using CoSENTLoss instead."
            )
        _validate_mini_batch_num_tokens(mini_batch_num_tokens)

        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.mini_batch_size = mini_batch_size
        self.mini_batch_num_tokens = mini_batch_num_tokens
        self.show_progress_bar = show_progress_bar

    def calculate_loss(
        self, reps: list[list[Tensor]], labels: Tensor | None = None, *, with_backward: bool = False
    ) -> Tensor:
        """Calculate the CoSENT loss, chunked over ``mini_batch_size`` rows of the comparison matrix.

        The logsumexp over the whole matrix is accumulated online, one block of rows at a time, together
        with its gradient wrt. the pair scores. That gradient is rescaled whenever the running total grows,
        so it ends up exact without ever holding more than one block in memory.
        """
        if labels is None:
            raise ValueError("CachedCoSENTLoss requires a float similarity score label for every input pair.")
        embeddings = [torch.cat(column) for column in reps]
        # (batch_size,), the only thing the loss depends on
        pair_scores = self.similarity_fct(embeddings[0], embeddings[1]) * self.scale
        scores = pair_scores.detach().float()
        batch_size = len(scores)

        # Start from e^0 = 1, the "1 +" inside the log
        total = scores.new_zeros(())
        grad = torch.zeros_like(scores)
        for begin in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Calculating loss",
            disable=not self.show_progress_bar,
        ):
            end = min(begin + self.mini_batch_size, batch_size)
            # block[r, c] = s(r) - s(c), for the rows r in [begin, end)
            block = scores[begin:end, None] - scores[None, :]

            # mask out irrelevant pairs so they are negligible after exp(), as in CoSENTLoss
            relevant = labels[begin:end, None] < labels[None, :]
            block = block - (~relevant) * 1e12

            new_total = torch.logaddexp(total, torch.logsumexp(block.flatten(), dim=0))
            if with_backward:
                # d(new_total)/d(block) = exp(block - new_total). Each entry adds its weight to its row's
                # score and subtracts it from its column's score.
                weights = torch.exp(block - new_total)
                grad *= torch.exp(total - new_total)
                grad[begin:end] += weights.sum(dim=1)
                grad -= weights.sum(dim=0)
            total = new_total

        if with_backward:
            pair_scores.backward(grad.to(pair_scores.dtype))
        return total

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        sentence_features = list(sentence_features)
        if len(sentence_features) != 2:
            raise ValueError(
                f"CachedCoSENTLoss expects (input_A, input_B) pairs, i.e. 2 inputs, got {len(sentence_features)}."
            )
        return self.forward_cached(sentence_features, labels)

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "similarity_fct": similarity_fct_name(self.similarity_fct),
            "mini_batch_size": self.mini_batch_size,
            "mini_batch_num_tokens": self.mini_batch_num_tokens,
        }

    @property
    def citation(self) -> str:
        return """
@article{10531646,
    author={Huang, Xiang and Peng, Hao and Zou, Dongcheng and Liu, Zhiwei and Li, Jianxin and Liu, Kay and Wu, Jia and Su, Jianlin and Yu, Philip S.},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    title={CoSENT: Consistent Sentence Embedding via Similarity Ranking},
    year={2024},
    doi={10.1109/TASLP.2024.3402087}
}
"""
