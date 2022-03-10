#//////////////////////////////////////////////////////////
#   question_classifier.py
#
#   Created on:      11-Oct-2021 12:55:00
#   Original Authors: Alexander Trebilcock, Ashka K. Abrarriadi, 
#   Joshua Sayce, Oluwadamini Akpotohwo, Wenqi Han, Zhaoyu Han
#
#   Specification:
#
#   question_classifier.py is the base file for the Team Late 
#   submission for the Question Classifier project. It:
#
#   - Identifies arguments
#   - Interprets config files
#   - Preprocesses data, splitting it into text and labels.
#   - Conducts generic model behaviours on module instances
#   - Aggregates ensemble model results (if appropriate)
#   - Handles formatting of feedback/errors from modules to user
#
#//////////////////////////////////////////////////////////

#Library Imports
import argparse
import configparser

import codecs
import re
import numpy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from bow import main as bow_main
from bilstm import main as bilstm_main

from embedding import main as embedding_main

from ffnn_classifier import trainModel as ffnn_trainModel
from ffnn_classifier import testModel as ffnn_testModel

#Basic Structual Definitions
model_sources = {'bow': bow_main, 'bilstm': bilstm_main}

def main():
    
    #Read Arguments, Config Files
    args = handleArguments()
    config = readConfig(args.config)
     
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
        lemmadata, tokens = tokeniseData(text, stopWords)        
        
    elif args.test:

        dataDir = config["Paths"]["path_train"]

    #Load data from file
    data = loadData(dataDir)

    #Preprocess data: splits strings into lists of their labels and text.
    text, targets = preprocessData(data)     
    
    #Tokenise: Takes raw texts, returns their lemma form and a unique token list
    lemmadata, tokens = tokeniseData(text, stopWords)        

    #Embed: Convert unique token list into word embeddings.
    embeddings, vocabulary = generateWordEmbeddings(tokens, config)

    #Encode: Convert data into numerical equivalents
    encodeddata = encodeData(lemmadata, vocabulary)

    #Construct model
    modelstring = config["Model"]["model"]
    modelconfig = readConfig(config["Paths"][modelstring+"_config"])
    
    if(modelstring == "bilstm"):
        model = model_sources['bilstm'](embeddings, modelconfig, class_num=config["Network Structure"]["output_dimensions"])    
    
    elif(modelstring == "bow"):
        model = [] #obtain bow as model-like
    
    else:
        raise Exception("Error: no valid model specified (specified'" + modelstring + "'.")

    #Ensemble results loop
    results = []
    for i in range(ensemble_size):
        
        if args.train:
            #Train selected model (BOW or BiLSTM) if "train" arg specified
            results.append(ffnn_trainModel(encodeddata, targets, model))

        elif args.test:
            #Test selected model (BOW or BiLSTM) if "test" arg specified

            testModel()

            #Classify data (accuracy/F1 scores) produced by model if "test" arg specified
            results.append(classifyModelOutput())
            
    aggregateResults(results)
    
    displayResults()


#Checks for the three required arguments - train or test, manually specify config, and config path.
def handleArguments():

     # check parsed arguements (as found in coursework pdf)
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True, help='Configuration file')
        parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
        parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')

        # file arguements
        args = parser.parse_args()
        return args


#Attempts to parse provided file as config.                
def readConfig(configFile):

    # load config file (as found in coursework pdf)
    config = configparser.ConfigParser()
    config.sections()
    config.read(configFile)

    return config

#Write back to a config file with data.
def writeConfig(configFile, data):

    config = configparser.ConfigParser()
    
    with open(configFile, 'w') as file:
        config.write(file)


#Attempts to load data from the config-specified source for "Training Set 5".
def loadData(directory):

    with open(directory, "r", encoding = 'latin-1') as f: # get data
        data = f.readlines()
        data = [line[:-1] for line in data]

    return data


#Split the data into tokens. Returns tokenised text strings, and dict of unique words.
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
    
    return documents, uniqueWords


#Splits data into strings, and their respective labels.
def preprocessData(data):

    text, targets = [], []
    
    for line in data:
        delimline = line.split(" ", 1)
        text.append(delimline[1])
        targets.append(delimline[0])    
    
    return [text, targets]
    
    
#Container for link to embedding.py's embedding methods. Returns embeddings, and vocab list.
def generateWordEmbeddings(data, config):
    
    return embedding_main(data, config)
    
#Converts data into their indexes of their vocabulary, and pads if appropriate.
def encodeData(lemmadata, vocabulary):    
    
    encoded = []
    #The length we pad each encoded string to. Length of biggest instance *1.5.
    padlength = int(max([len(i) for i in lemmadata])*1.5) #Chosen arbitrarily.
    
    for sent in lemmadata:
        encode = []
        for word in sent:
            if word in vocabulary:
                encode.append(int(vocabulary.index(word)))
            else:
                encode.append(int(len(vocabulary)-1))
        
        encode += [0] * (padlength - len(encode))
        encoded.append(encode) 
        
    return encoded

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
