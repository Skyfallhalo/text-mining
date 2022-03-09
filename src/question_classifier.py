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
from bilstm import main as bilstm_main
from embedding import pretrained_embedding
from embedding import random_embedding
from ffnn_classifier import trainModel as ffnn_trainModel
from ffnn_classifier import testModel as ffnn_testModel
import codecs
import numpy
import re

#Local Imports


#Basic Structual Definitions
model_sources = {'bow': bow_main, 'bilstm': bilstm_main}

def main():
    
    #Read Arguments
    args = handleArguments()
    
    #Read Config Files 
    config = readConfig(args.config)
    
    outputDimensions = 1
    modelconfig = readConfig(config["Paths"]["bilstm_config"])
        
    stopWords = loadData(config["Paths"]["stop_words"])
    ensemble_size = int(config["Model"]["ensemble_size"]) 

    #Retrieve 
    if args.train:

        dataDir = config["Paths"]["path_train"]
        devDir = config["Paths"]["path_dev"]
        
        dev = loadData(devDir)

        #Preprocess data (stopwords, lemmatising)
        text, targets = preprocessData(dev)         
        
        #Tokensive and gen. word embeddings (RandomInit, Pre-trained), if "train" arg specified
        dev = tokeniseData(dev, stopWords)        
        
    elif args.test:

        dataDir = config["Paths"]["path_train"]

    data = loadData(dataDir)

    #Preprocess data (stopwords, lemmatising)
    text, targets = preprocessData(data)     
    
    #Tokensive
    tokens = tokeniseData(text, stopWords)        

    #Gen. word embeddings (RandomInit, Pre-trained), if "train" arg specified
    vocabulary, embeddings = generateWordEmbeddings(tokens, config)

    #Construct model
    model = model_sources['bilstm'](embeddings, modelconfig, class_num=outputDimensions)    

    results = []

    for i in range(ensemble_size):
        
        if args.train:
            #Train selected model (BOW or BiLSTM) if "train" arg specified
            results.append(ffnn_trainModel(data, model))

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



def load_file(path):
    quest_text = ''
    question = codecs.open(path, 'r', encoding = 'ISO-8859-1')
    question_lines = question.readlines()
    for line in question_lines:
        quest_text = quest_text + line
    question.close()
    return quest_text


#Split the data into tokens.
def tokeniseData(data,stopwords):
    text = data
    stopWords = stopwords
    #text = text.split("\n")
    text = text[:-1]
    #stopWords = stopWords.split("\n")
    
    
    documents = []
    for sen in text:
        # Remove all the special characters
        document = re.sub(r'\W', ' ', sen)
        
        # Converting to Lowercase
        document = document.lower()
        document = document.split()
        document = document[1:]
        
        i = 0
        for word in document:
            if word in stopWords:
                temp = document[1:]
                for i in range(len(temp)):
                    if temp[i] == word:
                        temp[i]=""
                        i+=1
                document = temp
                
        document = list(filter(None, document))
        documents.append(document)
    
    
    listofWords=""
    for doc in documents:
        for word in doc:
            listofWords = listofWords + " " + word
            
    dictionary = {}
    tempList = listofWords.split()
    for item in tempList:
        #print(item)
        if item in dictionary:
            dictionary[item] += 1
        else:
            dictionary.update({item: 1})
          
        
    k = 0
    newDict = {}
    uniqueWords=[]
    for (key,value) in dictionary.items():
        if value>k:
            newDict[key] = value
            uniqueWords.append(key)
    
    return uniqueWords

#Removes stopwords, lemma-izes, etc. according to config-specified rules.
def preprocessData(data):

    text, targets = [], []
    
    for line in data:
        delimline = line.split(" ", 1)
        text.append(delimline[1])
        targets.append(delimline[0])    
    
    return [text, targets]
    
    
#Uses either random, or pre-trained method to generate vector of word embeddings.
def generateWordEmbeddings(data, config):
    
    text, targets = [], []

    if(config["Embeddings"]["use_pretrained"]):
        vocab = config["Embeddings"]["path_vocabulary"]
        emb = config["Embeddings"]["path_pretrained"]
        return pretrained_embedding(vocab, emb)
    else:
        return random_embedding(data)
    
    
#Calls external model (as specified by config) with data, recieves returned data, saves results.   
def trainModel(data):

    model = model_sources['bilstm']()
    return ffnn_trainModel(data, model)
    
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
