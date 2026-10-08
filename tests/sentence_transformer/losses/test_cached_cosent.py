from __future__ import annotations

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import (
    AnglELoss,
    CachedCoSENTLoss,
    CoSENTLoss,
    MatryoshkaLoss,
)
from sentence_transformers.util import pairwise_angle_sim
from tests.sentence_transformer.losses.utils import assert_trained, disable_dropout, gradients

SENTENCES_A = ["a cat sits", "the sun is out", "he drove", "a dog barks", "it rains", "she reads", "birds sing"]
SENTENCES_B = ["a cat rests", "it is sunny", "she walked", "a cat meows", "it is dry", "he writes", "birds fly"]
# Includes ties, which CoSENT does not compare against each other
LABELS = [0.9, 0.8, 0.1, 0.4, 0.0, 0.4, 0.7]


def _split(embeddings: torch.Tensor, mini_batch_size: int) -> list[torch.Tensor]:
    """Split into the leaf mini-batch embeddings that the cached forward pass hands to calculate_loss."""
    return [mb.detach().clone().requires_grad_() for mb in embeddings.split(mini_batch_size)]


@pytest.mark.parametrize("mini_batch_size", [1, 2, 3, 7, 50])
@pytest.mark.parametrize("use_angle", [False, True])
def test_cached_cosent_calculate_loss_matches_cosent(
    stsb_bert_tiny_model: SentenceTransformer, mini_batch_size: int, use_angle: bool
) -> None:
    """The chunked loss stage must give exactly the loss and embedding gradients of CoSENTLoss
    (or AnglELoss for pairs), for any mini_batch_size, whether or not it divides the batch size."""
    torch.manual_seed(0)
    embeddings = [torch.randn(len(LABELS), 8), torch.randn(len(LABELS), 8)]
    labels = torch.tensor(LABELS)

    if use_angle:
        reference = AnglELoss(stsb_bert_tiny_model)
        cached = CachedCoSENTLoss(
            stsb_bert_tiny_model, similarity_fct=pairwise_angle_sim, mini_batch_size=mini_batch_size
        )
    else:
        reference = CoSENTLoss(stsb_bert_tiny_model)
        cached = CachedCoSENTLoss(stsb_bert_tiny_model, mini_batch_size=mini_batch_size)

    leaves = [e.clone().requires_grad_() for e in embeddings]
    expected = reference.compute_loss_from_embeddings(leaves, labels)
    expected.backward()

    reps = [_split(e, mini_batch_size) for e in embeddings]
    loss = cached.calculate_loss(reps, labels, with_backward=True)

    assert not loss.requires_grad
    torch.testing.assert_close(loss, expected.detach())
    for leaf, column in zip(leaves, reps):
        torch.testing.assert_close(torch.cat([mb.grad for mb in column]), leaf.grad)

    # Without the backward pass, e.g. during evaluation, only the loss is computed
    with torch.no_grad():
        torch.testing.assert_close(cached.calculate_loss(reps, labels), expected.detach())


@pytest.mark.parametrize("mini_batch_size", [2, 3, 7])
def test_cached_cosent_matches_cosent(stsb_bert_tiny_model: SentenceTransformer, mini_batch_size: int) -> None:
    """End to end through the model, mini_batch_size only bounds memory: the loss and the model
    gradients must match CoSENTLoss on the whole batch."""
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()
    labels = torch.tensor(LABELS)

    def loss_and_grads(loss_fn: torch.nn.Module) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model.zero_grad()
        features = [model.preprocess(SENTENCES_A), model.preprocess(SENTENCES_B)]
        loss_value = loss_fn(features, labels)
        loss_value.backward()
        return loss_value.detach(), gradients(model)

    cached_loss, cached_grads = loss_and_grads(CachedCoSENTLoss(model, mini_batch_size=mini_batch_size))
    plain_loss, plain_grads = loss_and_grads(CoSENTLoss(model))

    assert_trained(cached_grads)
    assert cached_loss.item() == pytest.approx(plain_loss.item(), rel=1e-4, abs=1e-5)
    for name, grad in cached_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=1e-5, msg=name)


def test_cached_cosent_matryoshka(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """MatryoshkaLoss decorates calculate_loss for cached losses, so the combination must match
    MatryoshkaLoss(CoSENTLoss(...))."""
    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()
    labels = torch.tensor(LABELS)
    dims = [128, 64, 32]

    def loss_and_grads(inner: torch.nn.Module) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model.zero_grad()
        features = [model.preprocess(SENTENCES_A), model.preprocess(SENTENCES_B)]
        loss_value = MatryoshkaLoss(model, inner, matryoshka_dims=dims)(features, labels)
        loss_value.backward()
        return loss_value.detach(), gradients(model)

    cached_loss, cached_grads = loss_and_grads(CachedCoSENTLoss(model, mini_batch_size=3))
    plain_loss, plain_grads = loss_and_grads(CoSENTLoss(model))

    assert_trained(cached_grads)
    assert cached_loss.item() == pytest.approx(plain_loss.item(), rel=1e-4, abs=1e-5)
    for name, grad in cached_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=1e-4, msg=name)


def test_cached_cosent_requires_pairs(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model
    loss = CachedCoSENTLoss(model, mini_batch_size=2)
    features = [model.preprocess(SENTENCES_A[:2]) for _ in range(3)]
    with pytest.raises(ValueError, match="expects \\(input_A, input_B\\) pairs"):
        loss(features, torch.tensor([0.1, 0.9]))


def test_cached_cosent_config(stsb_bert_tiny_model: SentenceTransformer) -> None:
    loss = CachedCoSENTLoss(stsb_bert_tiny_model, scale=10.0, mini_batch_size=16)
    assert loss.get_config_dict() == {
        "scale": 10.0,
        "similarity_fct": "pairwise_cos_sim",
        "mini_batch_size": 16,
        "mini_batch_num_tokens": None,
    }
