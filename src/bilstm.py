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
import torch.nn.functional as F
import torch.optim as optim


class BiLSTM(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.embedding = embedding
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 2 for bidirection

    def forward(self, seq):
        # # set initial states
        # h0 = torch.zeros(self.num_layers * 2, seq.size(0), self.hidden_size).to(self.device)  # 2 for bidirection
        # c0 = torch.zeros(self.num_layers * 2, seq.size(0), self.hidden_size).to(self.device)
        seq = self.embedding(seq)
        # out, _ = self.lstm(seq, (h0, c0))
        out, _ = self.lstm(seq)
        out = self.fc(out[:, -1, :])
        return out


def main(*args):
    # readConfig(args)
    return BiLSTM()


# Input: Config directory passed from question_classifier.py
# Task: Populate config values by reading config.ini
# Output: config.
def readConfig(configFile):
    print("Debug")    


if __name__ == "__main__":
    main(*sys.argv[1:])
    sys.exit(0)
