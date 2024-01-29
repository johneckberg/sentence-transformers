import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
from .. import util


class CoSENTLoss(nn.Module):
    """
    This class implements CoSENT (Consistent Sentence Embedding via Similarity Ranking) loss.
    It expects that each of the InputExamples consists of a pair of texts and a float valued label, representing
    the expected similarity score between the pair.


    It computes the following loss function:

    loss = logsum(1+exp(s(k,l)-s(i,j))+exp...), where (i,j) and (k,l) are any of the input pairs in the batch such that the expected
    similarity of (i,j) is greater than (k,l). The summation is over all possible pairs of input pairs in the batch that match this condition.

    For further details, see: https://kexue.fm/archives/8847


    :param model: SentenceTransformerModel
    :param similarity_fct: Function to compute the PAIRWISE similarity between embeddings. Default is util.pairwise_cos_sim.
    :param scale: Output of similarity function is multiplied by scale value. Represents the inverse temperature.


    Requirements:
        - Sentence pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].


    Inputs:
    +--------------------------------+------------------------+
    | Texts                          | Labels                 |
    +================================+========================+
    | (sentence_A, sentence_B) pairs | float similarity score |
    +--------------------------------+------------------------+


    Example::

        from sentence_transformers import SentenceTransformer, losses
        from sentence_transformers.readers import InputExample

        model = SentenceTransformer('bert-base-uncased')
        train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=1.0),
                InputExample(texts=['My third sentence', 'Unrelated sentence'], label=0.3)]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.CoSENTLoss(model=model)
    """

    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.pairwise_cos_sim):
        super(CoSENTLoss, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.scale = scale

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        scores = self.similarity_fct(embeddings[0], embeddings[1])
        scores = scores * self.scale
        scores = scores[:, None] - scores[None, :]

        # label matrix indicating which pairs are relevant
        labels = labels[:, None] < labels[None, :]
        labels = labels.float()

        # mask out irrelevant pairs so they are negligible after exp()
        scores = scores - (1 - labels) * 1e12

        # append a zero as e^0 = 1
        scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
        loss = torch.logsumexp(scores, dim=0)

        return loss

    def get_config_dict(self):
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}