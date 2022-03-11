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
import codecs
import configparser
import random
import re

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, random_split

from bilstm import main as bilstm_main
from bow import main as bow_main
from embedding import main as embedding_main
# from ffnn_classifier import testModel as ffnn_testModel
# from ffnn_classifier import trainModel as ffnn_trainModel
from ffnn_classifier_bkp import testModel as ffnn_testModel
from ffnn_classifier_bkp import trainModel as ffnn_trainModel

import matplotlib.pyplot
import matplotlib.ticker

#Basic Definitions
defaultConfig = '../data/config.ini'
model_sources = {'bow': bow_main, 'bilstm': bilstm_main}

#Set random seeds
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():

    #Read Arguments, Config Files
    args = handleArguments()
    config = readConfig(args.config)

    stopWords = loadData(config["Paths"]["stop_words"])
    ensembleSize = int(config["Model"]["ensemble_size"])

    #Retrieve
    if args.train:

        trainModel(config, ensembleSize, stopWords)

    elif args.test:

        # Obtain ensemble's results of interrogating the specified NN.
        results, results_ens = testModel(config, ensembleSize, stopWords)

        displayResults(results, results_ens)


#Checks for the three required arguments - train or test, manually specify config, and config path.
def handleArguments():

     # check parsed arguements (as found in coursework pdf)
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default=defaultConfig, type=str, help='Configuration file')
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
    data = []
    with open(directory, "r") as f:# get data
        for line in f:
            data.append(line.strip())

    return data


#Split the data into tokens. Returns tokenised text strings, and dict of unique words.
def tokeniseData(data, stopwords, min_occurence):
    text = data
    stopWords = stopwords
    #text = text.split("\n")
    #stopWords = stopWords.split("\n")

    documents = []
    for sen in text:
        # Remove all the special characters
        document = re.sub(r'\W', ' ', sen)

        # Converting to Lowercase
        document = document.lower()
        document = document.split()

        document = list(filter(lambda x: x not in stopWords, document))
        documents.append(document)

    dictionary = {}
    for doc in documents:
        for word in doc:
            if word not in dictionary:
                dictionary[word] = 0
            dictionary[word] += 1

    uniqueWords = []
    for key, value in dictionary.items():
        if value > min_occurence:
            uniqueWords.append(key)

    return documents, ["#unk#"] + uniqueWords


#Splits data into strings, and their respective labels.
def preprocessData(data):

    text, targets = [], []

    for line in data:
        delimline = line.split(" ", 1)
        text.append(delimline[1])
        targets.append(delimline[0])

    return text, targets


#Container for link to embedding.py's embedding methods. Returns embeddings, and vocab list.
def generateWordEmbeddings(data, config):

    return embedding_main(data, config)

#Converts data into their indexes of their vocabulary, and pads if appropriate.
def encodeData(lemmadata, vocabulary, padlength):

    encoded = []
    #The length we pad each encoded string to. Length of biggest instance *1.5.
    # padlength = int(max([len(i) for i in lemmadata])*1.5) #Chosen arbitrarily.

    unk_idx = vocabulary.index("#unk#")
    for sent in lemmadata:
        encode = [0] * padlength
        for i in range(len(sent)):
            if i == padlength:
                break
            word = sent[i]
            if word in vocabulary:
                encode[i] = vocabulary.index(word)
            else:
                encode[i] = unk_idx

        encode += [0] * (padlength - len(encode))
        encoded.append(encode)

    return encoded


def generateDatasets(X, y, ensemble_size, batch_size, train_size, min_split_size):

    if ensemble_size > 1:
        # Randomly split subsets
        split_size = min_split_size if int(len(X) / ensemble_size) < min_split_size else int(
            len(X) / ensemble_size)
        sub_idx = np.random.choice(range(len(X)), size=split_size)
        X_sub = [X[i] for i in sub_idx]
        y_sub = [y[i] for i in sub_idx]
        dataset = LateDataset(X_sub, y_sub)
        train_ds, dev_ds = random_split(dataset, [len(X_sub) - int(len(X_sub) * 0.1), int(len(X_sub) * 0.1)])
    else:
        X_train = X[:train_size]
        X_dev = X[train_size:]
        y_train = y[:train_size]
        y_dev = y[train_size:]
        train_ds = LateDataset(X_train, y_train)
        dev_ds = LateDataset(X_dev, y_dev)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=batch_size)

    return train_dl, dev_dl

#Contains the prepatory code for training
def trainModel(config, ensembleSize, stopWords):

    #Load misc. config
    classifierconfig = readConfig(config["Paths"]["classifier_config"])

    #Load data from file
    trainData = loadData(config["Paths"]["path_train"])
    devData = loadData(config["Paths"]["path_dev"])

    data = trainData + devData

    #Preprocess data: splits strings into lists of their labels and text.
    text, targets = preprocessData(data)

    #Tokenise: Takes raw texts, returns their lemma form and a unique token list
    lemmadata, tokens = tokeniseData(text, stopWords, int(config["Model"]["vocab_min_occurrence"]))

    #Embed: Convert unique token list into word embeddings.
    embeddings, vocabulary = generateWordEmbeddings(tokens, config)

    #Encode: Convert data into numerical equivalents
    encodeddata = encodeData(lemmadata, vocabulary, int(config["Model"]["sentence_max_length"]))

    #Get the list of unique classes of questions, the length of classes is supposed to be 50
    classes = list(set(targets))
    classes.sort()
    indexed_targets = [classes.index(x) for x in targets]

    #Construct model
    modelstring = config["Model"]["model"]
    modelconfig = readConfig(config["Paths"][modelstring+"_config"])

    #Ensemble results loop
    for i in range(ensembleSize):

        if modelstring == "bilstm" or modelstring == "bow":
            model = model_sources[modelstring](embeddings, modelconfig, len(classes))
            model.to(device)

        else:
            raise Exception("Error: no valid model specified (specified'" + modelstring + "'.")

        # Split training and development data and generate data loaders
        train_dl, dev_dl = generateDatasets(encodeddata, indexed_targets, ensembleSize,
                                            int(config["Network Structure"]["batch_size"]),
                                            len(trainData),
                                            int(config["Model"]["ensemble_min_split_size"]))

        #Train selected model (BOW or BiLSTM) if "train" arg specified
        print("Training for Model {0} of {1}".format(i, ensembleSize))
        model = ffnn_trainModel(train_dl, dev_dl, model,
                                numEpochs=int(classifierconfig["Model Settings"]["epoch"]),
                                lr=float(classifierconfig["Hyperparameters"]["lr_param"]))

        #Save the results of model's training to file.
        torch.save({
            "model_state_dict": model.state_dict()
        }, "{0}model.{1}.{2}.pt".format(config["Paths"]["path_model_prefix"], config["Model"]["model"], i))

    #Save word embeddings, vocab list, and class/label reference to file. 
    torch.save({
        "word_embedding_state_dict": embeddings.state_dict(),
        "vocab_list": vocabulary,
        "classes": classes
    }, config["Paths"]["path_cache"])    
    
def testModel(config, ensembleSize, stopWords):
    
    #Load data from checkpoint
    checkpoint = torch.load(config["Paths"]["path_cache"])

    vocabulary = checkpoint["vocab_list"]
    classes = checkpoint["classes"]

    #Embed: Load previously generated word embedding map.
    embeddings = nn.Embedding(checkpoint["word_embedding_state_dict"]["weight"].size(dim=0),
                              checkpoint["word_embedding_state_dict"]["weight"].size(dim=1))
    embeddings.load_state_dict(checkpoint["word_embedding_state_dict"])

    
    # Load data.
    testData = loadData(config["Paths"]["path_test"])

    # Preprocess data: splits strings into lists of their labels and text.
    text, targets = preprocessData(testData)

    # Tokenise: Takes raw texts, returns their lemma form and a unique token list
    lemmadata, _ = tokeniseData(text, stopWords, int(config["Model"]["vocab_min_occurrence"]))

    # Encode: Convert data/classes into numerical equivalents
    encodeddata = encodeData(lemmadata, vocabulary, int(config["Model"]["sentence_max_length"]))
    indexed_targets = [classes.index(x) for x in targets]

    # Convert data/labels to tensor format
    X_test = torch.LongTensor(encodeddata[:len(testData)])
    y_test = torch.LongTensor(indexed_targets[:len(testData)])

    results = []

    for i in range(ensembleSize):

        # Load model's training data
        try:
            filename = "{0}model.{1}.{2}.pt".format(config["Paths"]["path_model_prefix"], config["Model"]["model"], i)
            checkpoint = torch.load(filename)
        except FileNotFoundError:  
            raise Exception("Error: no model found '" + filename + "'. Was the model trained with a different ensemble count?")
        
        # Construct model instance (new for ensemble iteration)
        modelstring = config["Model"]["model"]
        modelconfig = readConfig(config["Paths"][modelstring + "_config"])
        model = model_sources[modelstring](embeddings, modelconfig, len(classes))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        #Test selected model (BOW or BiLSTM) if "test" arg specified
        results.append(ffnn_testModel(X_test, y_test, model))   
        
    #Aggregate results of ensemble
    y_pred_ens = aggregateResults(config, results)
    
    #return results, y_pred_ens, 
    
    #Classify data (accuracy/F1 scores) produced by model if "test" arg specified
    plaintext_lab = classifyModelOutput(y_test, y_pred_ens, classes)
    
    #Return plaintext results for easier graphing. Format: Question, Prediction, Actual Label.
    p_results = [[classes[encodedlabel] for encodedlabel in result.tolist()] for result in results]
    p_results = [[[val, p_results[i][j], targets[j]] for j, val in enumerate(text)] for i, result in enumerate(results)]
    
    p_results_ens = [classes[encodedlabel] for encodedlabel in y_pred_ens.tolist()]
    p_results_ens = [[val, p_results_ens[i], targets[i]] for i, val in enumerate(text)]    
    
    return p_results, p_results_ens

#Attempts to run FF-NN with data, recieves returned data, and saves results.
def classifyModelOutput(y, y_pred, classes):
    
    #Create class list of only classes that are relevant.
    idx_list = torch.ones(len(classes))
    for n in y.unique():
        idx_list[n.item()] = 0
    for n in y_pred.unique():
        idx_list[n.item()] = 0
    target_names = classes.copy()
    for i in torch.flip(torch.nonzero(idx_list, as_tuple=False), dims=[0]).squeeze():
        del target_names[i]
    
    #Print basic classification report statistics
    print(classification_report(y.cpu(), y_pred.cpu(), target_names=target_names, zero_division=0))
    print("accuracy:", accuracy_score(y.cpu(), y_pred.cpu()))
    print("micro F1 score:", f1_score(y.cpu(), y_pred.cpu(), average='micro'))
    print("macro F1 score:", f1_score(y.cpu(), y_pred.cpu(), average='macro'))
    print("weighted F1 score:", f1_score(y.cpu(), y_pred.cpu(), average='weighted'))
    
    return target_names

#Takes the map of tensor results, and uses an ensemble model to generate a single set of classifications.
def aggregateResults(config, results):
    
    #Results: array of ensemble predictions
    #y_test: Gold standard labels
    
    results = torch.stack(results) 
    y_pred, _ = torch.mode(results, 0)
    return y_pred


def displayResults(results, results_ens):
    
    #Using MatPlotLib
    fig, ax = matplotlib.pyplot.subplots() # Make a new, large subplot 


class LateDataset(Dataset):
    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return torch.tensor(self.text[idx], dtype=torch.int32), self.label[idx]

if __name__ == "__main__":

    main()
    exit()
