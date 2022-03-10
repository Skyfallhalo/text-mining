#//////////////////////////////////////////////////////////
#   bow.py
#   Created on:      01-Mar-2022 14:30:00
#   Original Author: J. Sayce
#   Specification:
#
#   base file for the 'bow' model, used by question_classifier.py.
#
#//////////////////////////////////////////////////////////
import torch
from torch import nn


class BOW(nn.Module):
    def __init__(self, embedding, class_num):
        super(BOW, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = embedding

        self.fc = nn.Linear(embedding.weight.size(dim=1), class_num)

    def forward(self, seq):
        # seq = self.embedding(seq).to(self.device)
        sentence_vectors = []  # finish list of bag of words sentence vectors

        for sentence in seq:
            nonzero_sentence = sentence[torch.nonzero(sentence).squeeze()]
            middle = self.embedding(nonzero_sentence)
            if middle.dim() == 1:
                middle = torch.unsqueeze(middle, 0)
            middle = middle.sum(dim=0)
            sentence_vectors.append(middle)

        out = torch.stack(sentence_vectors)
        out = self.fc(out)
        return out


def main(embedding, config, class_num):
    return BOW(embedding, class_num)

