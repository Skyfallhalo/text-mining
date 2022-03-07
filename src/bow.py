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

    def __init__(self, data_size, embedding_size):

        super(BOWClassifier, self).__init__()
        self.lin = nn.Linear(data_size, embedding_size)

    def forward(self, x):

        return F.softmax(self.lin(x))

def make_bow_vector(sentence, embedding):
    # create a vector of zeros of vocab size = len(word_to_idx)
    vec = torch.zeros(len(embedding))

    for word in sentence:

        vec[embedding[word]]+=1

    return vec.view(1, -1)

def make_target(label, embedding):

    return torch.LongTensor([embedding[label]])



def main(data_size, embedding_size):

    return BOWClassifier(data_size, embedding_size)
    
#Input: Config directory passed from question_classifier.py
#Task: Populate config values by reading config.ini
#Output: config.
def readConfig(configFile): 


    print("Debug")    
    
