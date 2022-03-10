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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BOWClassifier(nn.Module):

    def __init__(self, embedding, hidden_dim, num_layers):

        super(BOWClassifier, self).__init__()
        
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = embedding

        self.fc_hidden = nn.Linear(hidden_dim,hidden_dim)
        self.bn_hidden = nn.BatchNorm1d(hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(hidden_dim, 1)
        
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x,target):

        bow_embedding = []
        for i in range(len(x)):
            lookup_tensor = Variable(torch.LongTensor(x[i])).cuda()
            embed = self.embedding(lookup_tensor)
            embed = embed.mean(dim=0)
            bow_embedding.append(embed)
        bow_embedding = torch.stack(bow_embedding)
    
        h = self.dropout(F.relu(self.bn_hidden(self.fc_hidden(bow_embedding))))
        h = self.fc_output(h)
    
        return self.loss(h[:,0],target), h[:,0]


def main(embedding, config, embedding_size):

    return BOWClassifier(embedding, int(config["Arguments"]["hidden_dim"]), int(config["Arguments"]["num_layers"]))
    
#Input: Config directory passed from question_classifier.py
#Task: Populate config values by reading config.ini
#Output: config.
def readConfig(configFile): 


    print("Debug")    
    
