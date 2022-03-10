# //////////////////////////////////////////////////////////
#   bilstm.py
#   Created on:      01-Mar-2022 14:30:00
#   Original Author: J. Sayce
#   Specification:
#
#   base file for the 'bilstm' model, used by question_classifier.py.
#
# //////////////////////////////////////////////////////////

import sys
import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, embedding, hidden_dim, num_layers, class_num):
        '''
        Initialise a BiLSTM model.

        :param embedding: The word embedding object
        :type embedding: Embedding
        :param hidden_dim: The number of features in the hidden state
        :type hidden_dim: int
        :param num_layers: Number of recurrent layers
        :type num_layers: int
        :param class_num: The number of output features in the final linear neural network
        :type class_num: int
        '''
        super(BiLSTM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.embedding = embedding
        embedding_dim = embedding.weight.size(dim=1)

        # Initialise BiLSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Initialise feed-forward neural network which is used to covert the dimension of hidden state feature to the out put dimension
        # self.fc = nn.Linear(hidden_dim * (2 if self.lstm.bidirectional else 1), class_num)  # 2 for bidirection
        self.fc = nn.Linear(hidden_dim, class_num)

    def forward(self, seq):
        '''
        Train model during training phase.

        This function defines the computation performed at every call. When the function is called,
        the BiLSTM model is used firstly to generate the final output of hidden state. Then using linear
        transformation, the dimension of hidden layer is transformed to the dimension of classes for classification.

        :param seq: A batch of input data
        :type seq: Tensor
        :return:
        '''
        # # set initial states
        # h0 = torch.zeros(self.num_layers * 2, seq.size(0), self.hidden_size).to(self.device)  # 2 for bidirection
        # c0 = torch.zeros(self.num_layers * 2, seq.size(0), self.hidden_size).to(self.device)
        # print(seq.shape)
        seq = self.embedding(seq).to(self.device)
        # print(seq.shape)
        # out, _ = self.lstm(seq, (h0, c0))
        out, (ht, ct) = self.lstm(seq)
        # out, _ = self.lstm(seq)
        # print(out.shape)
        # print(ht.shape)
        # print(ct.shape)
        # print(out[:, -1, :].shape)
        # print((ht[0, :, :] + ht[1, :, :]).shape)
        # out = self.fc(out[:, -1, :self.hidden_size] + out[:, 0, self.hidden_size:])
        out = self.fc(ht[0, :, :] + ht[1, :, :])
        # out = self.fc(out[:, -1, :])
        # print(out.shape)
        # print("bilstm")
        # print(torch.max(out, 1)[1], torch.max(F.softmax(out, dim=1), 1)[1])
        return out


def main(embedding, config, class_num):
    # readConfig(args)
    return BiLSTM(embedding, int(config["Arguments"]["hidden_dim"]), int(config["Arguments"]["num_layers"]), class_num)


# Input: Config directory passed from question_classifier.py
# Task: Populate config values by reading config.ini
# Output: config.
def readConfig(configFile):
    print("Debug")    


if __name__ == "__main__":
    main(*sys.argv[1:])
    sys.exit(0)
