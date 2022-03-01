#//////////////////////////////////////////////////////////
#   question_classifier.py
#   Reinterpretation of various unique implementations of 
#   the "wc" command
#   Created on:      11-Oct-2021 12:55:00
#   Original Author: J. Sayce
#   Specification:
#
#   question_classifier.py is the base file for the Team Late 
#   submission for the Question Classifier project. It:
#
#   - Handles arguments
#   - Validates config files (and complains if they're wrong)
#   - Loads appropriate modules (preprocessor, embedder...) into generic steps
#   - Conducts generic model behaviours on module instances
#   - Aggregates ensemble model results (if appropriate)
#   - Handles formatting of feedback/errors from modules to user
#
#//////////////////////////////////////////////////////////

#Library Imports
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

def train(configFile):

    configLoader(configFile)

def test(configFile):

    configLoader(configFile)

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
    

#Local Imports


#Basic Structual Definitions
model_sources = {
    'bow' : (new bow()),
    'bilstm' : (new bilstm())
}

def main(*args):
    
    #Read Arguments
    args_ok = handleArguments()
    
    #Read Config Files 
    config_ok = readConfig()
    
    while(args_ok and config_ok):

        #Retrieve 
        data = loadData()
        
        #Tokensive and gen. word embeddings (RandomInit, Pre-trained), if "train" arg specified
        data = tokeniseData(data)        
        
        #Preprocess data (stopwords, lemmatising)
        data = preprocessData(data) 
          
        for i in range(ensemble_count):
            
            #Train selected model (BOW or BiLSTM) if "train" arg specified
            if train_model:
                trainModel()
            
            #Test selected model (BOW or BiLSTM) if "test" arg specified
            if test_model:
                testWithModel()
            
            #Classify data (accuracy/F1 scores) produced by model if "test" arg specified
            if test_model:
                classifyModelOutput()
    
    aggregateResults()
    
    displayResults()

def handleArguments():
    return true
    
    
def readConfig():  
    if model_name not in model_sources:
        raise Exception("Error: unknown model specified: '" + model_name + "'.") 
    return true


#Attempts to load data from the config-specified source for "Training Set 5".
def loadData():
    return ("DESC:other Lorem ?", "DESC:other Ipsum")


#Split the data into tokens.
def tokeniseData(data):
    for line in data:
        line = line.split()
    return data


#Removes stopwords, lemma-izes, etc. according to config-specified rules.
def preprocessData(data):
    for line in data:
        line = line.replace('stopword', '')
    return data
    
    
#Uses either random, or pre-trained method to generate vector of word embeddings.
def generateWordEmbeddings():
    return [('lorem', 1), ('ipsum', 0)]
   
    
#Calls external model (as specified by config) with data, recieves returned data, saves results.   
def trainModel():
    for line in data:
        line = line.split()
    return data
    
#Attempts to run BOW or BiLSTM with data, recieves returned data, and saves results.    
def testWithModel():
    print("Debug!")
    
    
#Attempts to run FF-NN with data, recieves returned data, and saves results.
def classifyModelOutput():
    print("Debug!")
    
    
def aggregateResults():
    print("Debug!")
   
    
def displayResults():
    print("Debug!")


if __name__ == "__main__":
    
    main(*sys.argv[1:])
    sys.exit(0)
