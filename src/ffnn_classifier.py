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

import numpy as np

import torch   
from torch.utils.data import Dataset, DataLoader

import torchtext   
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


#inference 
#import spacy

class LateDataset(Dataset):
    def __init__(self, text, label):
        self.text = text
        self.label = label
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.text[idx][0].astype(np.int32)), self.label[idx], self.text[idx][1]

#Accuracy Metric
def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc
    
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)    

def trainModel(data, labels, model):
           
    batchSize = 128  
    numEpochs = 10
    bestLoss = float('inf')       
    
    #Data Preparation
    data = [np.array(x) for x in data]
    dataset = LateDataset(data, labels)
    data_train = DataLoader(dataset = dataset, batch_size = batchSize, shuffle =False)
    
    #Define the Optim. (Stochastic G.D.) and Loss metric
    optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.BCELoss()    

    #CUDA
    #Cuda algorithms
    torch.backends.cudnn.deterministic = True    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      
    model = model.to(device)
    criterion = criterion.to(device)    
    
    #Main loop
    epoch_loss, epoch_acc = 0, 0
     
    for epoch in range(numEpochs):

        for x, y, l in data_train:
            
            model.zero_grad()
         
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
         
            tag_scores = model(data)
            
            #Compute loss/accuracy 
            loss = loss_function(tag_scores, targets)
            acc = binary_accuracy(predictions, batch.Label)   
            
            #Backpropogate and optimise
            loss.backward()
            optimizer.step()        
        
            #Loss and accuracy
            epoch_loss += loss.item()  
            epoch_acc += acc.item()           
        
    train_loss, train_acc = epoch_loss / len(iterator), epoch_acc / len(iterator)

    ##Save a local best to file
    #if train_loss < bestLoss:
    #    bestLoss = valid_loss
    #    torch.save(model.state_dict(), 'saved_weights.pt')
    #
    #print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    #print(f'\tTest Loss: {valid_loss:.3f} |  Test Acc: {valid_acc*100:.2f}%')   
    
    ##load weights
    #path='saved_weights.pt'
    #model.load_state_dict(torch.load(path));
    #model.eval();    
    
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