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

    def __init__(self, embedding, num_labels):

        super(BOWClassifier, self).__init__()

        self.embedding = embedding
        embedding_dim = embedding.weight.size(dim=1)
        self.linear = nn.Linear(embedding_dim, num_labels)

    def forward(self, x):

        bow_vector= []

        for i in x: # find vec_bow(x)

            i = i[torch.nonzero(i).squeeze()] # remove padding words
            lookup_tensor = Variable(i)
            vec = self.embedding(lookup_tensor)

            if vec.dim() == 1: # add padding vectors back

                vec = torch.unsqueeze(vec, 0)

            vec = vec.sum(dim=0)
            bow_vector.append(vec)
            
        bow_vector = torch.stack(bow_vector)

        return F.log_softmax(self.linear(bow_vector), dim=1)


def main(embedding, config, num_labels):

    return BOWClassifier(embedding, num_labels)
    
#Input: Config directory passed from question_classifier.py
#Task: Populate config values by reading config.ini
#Output: config.
def readConfig(configFile): 


    print("Debug")
