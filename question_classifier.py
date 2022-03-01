import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import configparser
import argparse

class BiLSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The BiLSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

def configLoader(configFile):

    # load config file (as found in coursework pdf)
    config = configparser.ConfigParser()
    config.sections()
    config.read(configFile)

    variables = []

    for i in config.sections():
        for j in config[i]:

            globals()[j] = config[i][j] # make global variable with variable name in config file

def bow():

    print("a")

def train(configFile):

    configLoader(configFile)

def test(configFile):

    variables = configLoader(configFile)
    for var in variables: # create variables using variable name and value

        exec("%s = %d" % (var[0],var[1]))

def main():
    # check parsed arguements (as found in coursework pdf)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
    parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')

    args = parser.parse_args()
    if args.train:
        #call train function
        train(args.config)
    elif args.test:
        #call test function
        test(args.config)

if __name__=="__main__":
    main()
    

