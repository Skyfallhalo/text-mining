#//////////////////////////////////////////////////////////
#   question_classifier.py
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
from bow import main as bow_main
from bilstm import BiLSTM as bilstm_main
from embedding import main as embedding_main
from ffnn_classifier import trainModel
from ffnn_classifier import testModel

#Local Imports


#Basic Structual Definitions
model_sources = {'bow': bow_main, 'bilstm': bilstm_main}

def main():
    
    #Read Arguments
    args = handleArguments()
    
    #Read Config Files 
    config = readConfig(args.config)
    
    stopWords = loadData(config["Paths"]["stop_words"])

    #Retrieve 
    if args.train:

        dataDir = config["Paths"]["path_train"]
        devDir = config["Paths"]["path_dev"]
        
        dev = loadData(devDir)

        #Tokensive and gen. word embeddings (RandomInit, Pre-trained), if "train" arg specified
        dev = tokeniseData(dev)        
        
        #Preprocess data (stopwords, lemmatising)
        dev = preprocessData(dev, stopWords) 

    elif args.test:

        dataDir = config["Paths"]["path_train"]

    data = loadData(dataDir)

    #Tokensive and gen. word embeddings (RandomInit, Pre-trained), if "train" arg specified
    data = tokeniseData(data)        

    #Preprocess data (stopwords, lemmatising)
    data = preprocessData(data, stopWords) 

    ensemble_size = int(config["Model"]["ensemble_size"])
    
    word_embeddings = embedding_main(data)

    results = []

    for i in range(ensemble_size):
        
        if args.train:
            #Train selected model (BOW or BiLSTM) if "train" arg specified

            results.append(trainModel(data))

        elif args.test:
            #Test selected model (BOW or BiLSTM) if "test" arg specified

            testModel()

            #Classify data (accuracy/F1 scores) produced by model if "test" arg specified
            results.append(classifyModelOutput())
            
    aggregateResults(results)
    
    displayResults()

def handleArguments():

     # check parsed arguements (as found in coursework pdf)
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True, help='Configuration file')
        parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
        parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')

        # file arguements
        args = parser.parse_args()
        return args
                
def readConfig(configFile):

    # load config file (as found in coursework pdf)
    config = configparser.ConfigParser()
    config.sections()
    config.read(configFile)

    return config

def writeConfig(configFile, data):

    config = configparser.ConfigParser()
    
    with open(configFile, 'w') as file:
        config.write(file)


#Attempts to load data from the config-specified source for "Training Set 5".
def loadData(directory):

    with open(directory, "r", encoding = 'latin-1') as f: # get data
        data = f.readlines()

    return data


#Split the data into tokens.
def tokeniseData(data):

    for line in data:
        line = line.split()
    return data


#Removes stopwords, lemma-izes, etc. according to config-specified rules.
def preprocessData(data, stopWords):

    for line in data:
        line = line.replace('stopword', '')
    return data
    
    
#Uses either random, or pre-trained method to generate vector of word embeddings.
def generateWordEmbeddings():

    return [('lorem', 1), ('ipsum', 0)]
   
    
#Calls external model (as specified by config) with data, recieves returned data, saves results.   
def trainModel(data):

    model = model_sources['bilstm']
    return trainModel(model)
    
#Attempts to run BOW or BiLSTM with data, recieves returned data, and saves results.    
def testModel(data):
    
    return testModel()
    
    
#Attempts to run FF-NN with data, recieves returned data, and saves results.
def classifyModelOutput():
    print("Debug!")
    
    
def aggregateResults():
    print("Debug!")
   
    
def displayResults():
    print("Debug!")


if __name__ == "__main__":
    
    main()
    exit()
