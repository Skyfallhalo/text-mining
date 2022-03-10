# //////////////////////////////////////////////////////////
#   bilstm.py
#   Created on:      01-Mar-2022 14:30:00
#   Original Author: J. Sayce
#   Specification:
#
#   base file for the 'bilstm' model, used by question_classifier.py.
#
# //////////////////////////////////////////////////////////

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, embedding, hidden_dim, num_layers, class_num):
        """
        Initialise a BiLSTM model.

        :param embedding: The word embedding object
        :type embedding: Embedding
        :param hidden_dim: The number of features in the hidden state
        :type hidden_dim: int
        :param num_layers: Number of recurrent layers
        :type num_layers: int
        :param class_num: The number of output features in the final linear neural network
        :type class_num: int
        """
        super(BiLSTM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.embedding = embedding
        embedding_dim = embedding.weight.size(dim=1)

        # Initialise BiLSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Initialise feed-forward neural network which is used to covert the dimension of
        # hidden state feature to the out put dimension
        self.fc = nn.Linear(hidden_dim, class_num)

    def forward(self, seq):
        """
        Train model during training phase.

        This function defines the computation performed at every call. When the function is called,
        the BiLSTM model is used firstly to generate the final output of hidden state. Then using linear
        transformation, the dimension of hidden layer is transformed to the dimension of classes for classification.

        :param seq: A batch of input data (encoded sentences)
        :type seq: Tensor
        :return:
        """
        # Get word vectors of each sentences
        seq = self.embedding(seq).to(self.device)

        # Forward pass through BiLSTM model and get the output
        out, (ht, ct) = self.lstm(seq)

        # Use the combination of hidden states of both forward direction
        # and backward direction to transform output
        out = self.fc(ht[0, :, :] + ht[1, :, :])
        return out


def main(embedding, config, class_num):
    return BiLSTM(embedding, int(config["Arguments"]["hidden_dim"]), int(config["Arguments"]["num_layers"]), class_num)
