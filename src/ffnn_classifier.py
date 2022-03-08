#//////////////////////////////////////////////////////////
#   ffnn_classifier.py
#   Created on:      01-Mar-2022 14:30:00
#   Original Author: J. Sayce
#   Specification:
#
#   base file for the feed-forward neural network classifier,
#   used by used by question_classifier.py.
#
#//////////////////////////////////////////////////////////

import sys
import random

import torch   
from torch.utils.data import Dataset, DataLoader

import torchtext   
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


#inference 
#import spacy

#Converts file into list of Examples
def getFromFile(directory, fields):
    examples = []
    with open(directory) as my_file:
        for line in my_file:
            delimline = line.split(" ", 1)            
            examples.append(torchtext.legacy.data.Example.fromlist((delimline[1], delimline[0]), fields))
    return examples
    
    
#Input: Config directory passed from question_classifier.py
#Task: Populate config values by reading config.ini
#Output: config.
def readConfig(configFile): 
    print("Debug")    
    
#Accuracy Metric
def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

def main(*args):
    
    #Get Config Params
    readConfig(args)
    
 
    
    #Static Hyperparams   
    embeddingDimensions = 100
    hiddenlayerDimensions = 2
    outputDimensions = 1
    layerCount = 2
    isBidirectional = True
    dropoutRate = 0 
    
    #Cuda algorithms
    torch.backends.cudnn.deterministic = True    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    
    #Define Data (Text, Label)
    dataText = torchtext.legacy.data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
    dataLabel = torchtext.legacy.data.LabelField(dtype = torch.float, batch_first=True)   
    fields = [('Text', dataText),('Label', dataLabel)]        
      
    #Get Train, Test dataset
    data_train = torchtext.legacy.data.Dataset(getFromFile('../data/train.txt', fields), fields = fields)
    data_test = torchtext.legacy.data.Dataset(getFromFile('../data/test.txt', fields), fields = fields)

    #Get Embeddings
    dataText.build_vocab(data_train,min_freq=3) #vectors = "../data/glove.txt"
    dataLabel.build_vocab(data_train)
    
    #Derived Hyperparams
    vocabularySize = len(dataText.vocab)
    
 
    
    #Model Construction
    model = questionClassifier(vocabularySize, embeddingDimensions, hiddenlayerDimensions,outputDimensions, layerCount, 
                       isBidirectional = isBidirectional, dropoutRate = dropoutRate)    
    
    #Initialize the pretrained embedding
    if(dataText.vocab.vectors):
        pretrained_embeddings = dataText.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)
    
    #Define the Optim. (Stochastic G.D.) and Loss metric
    optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.BCELoss()    
    
    #CUDA
    model = model.to(device)
    criterion = criterion.to(device)     
       
    #Main loop
    for epoch in range(numEpochs):
         
        #Train for this cycle
        train_loss, train_acc = trainModel(model, train_iterator, optimiser, criterion)
        
        #Test for this cycle
        valid_loss, valid_acc = evalModel(model, test_iterator, criterion)
        
        #Save a local best to file
        if valid_loss < bestLoss:
            bestLoss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pt')
        
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\tTest Loss: {valid_loss:.3f} |  Test Acc: {valid_acc*100:.2f}%')    
    
    
    #load weights
    path='saved_weights.pt'
    model.load_state_dict(torch.load(path));
    model.eval();

def trainModel(data, model):
           
    batchSize = 128  
    numEpochs = 10
    bestLoss = float('inf')       
    
    #Define the Optim. (Stochastic G.D.) and Loss metric
    data_train = DataLoader(dataset = data, batch_size = batchSize, shuffle =False)
    optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.BCELoss()    

    #CUDA
    #Cuda algorithms
    torch.backends.cudnn.deterministic = True    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      
    model = model.to(device)
    criterion = criterion.to(device)    
    
    #Main loop
    for epoch in range(numEpochs):

        epoch_loss, epoch_acc = 0, 0
        
        model.train()  
        
        for bidx, batch in tqdm(enumerate(data_train)):
            
            #Reset grads
            optimizer.zero_grad()   
            
            #Prepare inputs
            text, text_lengths = batch.Text   
            
            #Forward pass
            predictions = model(text)
            
            #Compute loss/accuracy and backpropogate
            loss = criterion(predictions, batch.Label)        
            acc = binary_accuracy(predictions, batch.Label)   
            loss.backward()       
            optimizer.step()      
            
            #loss and accuracy
            epoch_loss += loss.item()  
            epoch_acc += acc.item()    
            
        train_loss, train_acc = epoch_loss / len(iterator), epoch_acc / len(iterator)

        #Save a local best to file
        if train_loss < bestLoss:
            bestLoss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pt')

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\tTest Loss: {valid_loss:.3f} |  Test Acc: {valid_acc*100:.2f}%')     
    
def testModel(model, iterator, criterion):
    
    epoch_loss, epoch_acc = 0, 0

    model.eval()
    
    #deactivates autograd
    with torch.no_grad():
    
        for batch in iterator:
        
            #Prepare inputs
            text, text_lengths = batch.Text
            
            #Forward pass (there's an error here)
            predictions = model(text)
            
            #Compute loss/accuracy, do not backpropogate
            loss = criterion(predictions, batch.Label)
            acc = binary_accuracy(predictions, batch.Label)
            
            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    
if __name__ == "__main__":
    main(*sys.argv[1:])
    sys.exit(0)