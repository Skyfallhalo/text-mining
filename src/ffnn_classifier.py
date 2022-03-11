#//////////////////////////////////////////////////////////
#   ffnn_classifier.py
#   Created on:      01-Mar-2022 14:30:00
#   Original Author: W. Han, J. Sayce
#   Specification:
#
#   base file for the feed-forward neural network classifier,
#   used by used by question_classifier.py.
#
#//////////////////////////////////////////////////////////

import sys
import random
import time

import numpy as np

import torch   
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

class LateDataset(Dataset):
    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return torch.tensor(self.text[idx], dtype=torch.int32), self.label[idx]
    
def trainModel(data_train, data_dev, model, numEpochs=10, lr=0.1):
           
    bestLoss = float('inf')
    
    #Define the Optim. (Stochastic G.D.) and Loss metric
    optimiser = optim.SGD(model.parameters(), lr=lr)
    loss_function = torch.nn.functional.cross_entropy

    #CUDA
    #Cuda algorithms
    torch.backends.cudnn.deterministic = True    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      
    model = model.to(device)  
    
    #Main loop
    trainLoss, trainAcc = 0, 0
     
    for epoch in range(numEpochs):
        epochStart = time.time()
 
        model.train()
        
        epochLoss, epochTotal = 0, 0
        
        for text, targets in data_train:
            
            optimiser.zero_grad()

            text = text.type(torch.LongTensor).to(device)
            targets = targets.to(device)            
         
            tag_scores = model(text)
            
            #Compute loss 
            loss = loss_function(tag_scores, targets)
            
            #Backpropogate and optimise
            loss.backward()
            optimiser.step()        
        
            #Epoch loss and accuracy
            epochLoss += loss.item() * targets.shape[0]   
            epochTotal += targets.shape[0]
                    
        trainLoss = epochLoss / epochTotal
        
        #Save a local best to file
        model.eval()
        
        correctClassifications = 0
        valueLoss = 0.0
        vTotal = 0
        
        for text, targets in data_dev:
            text = text.type(torch.LongTensor).to(device)
            targets = targets.to(device)
            
            tag_scores = model(text)
            
            #Compute loss
            loss = loss_function(tag_scores, targets)
            
            #Epoch loss and accuracy       
            valueLoss += loss.item() * targets.shape[0]
            vTotal += targets.shape[0]
            
            goldLabels = torch.max(tag_scores, 1)[1]
            correctClassifications += (goldLabels == targets).sum().item()
            
        vLoss = valueLoss / vTotal
        vAccuracy = correctClassifications / vTotal
        
        #Finish and print
        epochTime = time.time() - epochStart

        printResults(epoch, numEpochs, trainLoss, vLoss, vAccuracy, epochTime)
        
    return model
    
def printResults(epoch, numEpochs, trainLoss, vLoss, vAccuracy, epochTime):
    print("Epoch [{:02}/{:02}]".format(epoch+1, numEpochs), end='\t')
    print("loss {:.4}".format(trainLoss), end='\t')
    print("val loss {:.4}".format(vLoss), end='\t')
    print("val_acc {:.4}".format(vAccuracy), end='\t')
    print("t time {:.4}".format(epochTime), end='\n')
    
    
def testModel(text, targets, model):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    epoch_loss, epoch_acc = 0, 0

    text = text.to(device)
    targets = targets.to(device)
    tag_scores = model(text)
    predictions = torch.max(tag_scores, 1)[1]
    
    return predictions

    
if __name__ == "__main__":
    main(*sys.argv[1:])
    sys.exit(0)