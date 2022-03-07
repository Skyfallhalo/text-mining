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
import torchtext   
import torch.nn as nn

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

class questionClassifier(nn.Module):

    def __init__(self, 
                 vocabularySize,
                 embeddingDimensions,  
                 hiddenlayerDimensions, 
                 outputDimensions, 
                 layerCount, 
                 isBidirectional, 
                 dropoutRate):
        
        super().__init__()          

        modelChosen = 'bilstm'
        
        #Embedding Layer (nn.Embedding)
        self.embedding = nn.Embedding(vocabularySize, embeddingDimensions)

        if(modelChosen == 'bow'): #BOW layer (nn.lin)
            self.lin = nn.Linear(vocabularySize, embeddingDimensions)
            
        elif (modelChosen == 'bilstm'): #(Bi)LSTM layer (nn.LSTM)
            
            self.lstm = nn.LSTM(embeddingDimensions, #Number of expected features in input x
                                hiddenlayerDimensions,  #Number of features in the hidden layers
                                layerCount,         #Number of layers of stacked BiLSTM instances.
                                bias=True,          #Applies bias values.
                                batch_first=True,   #Structure of output (b, s, f) vs. (s, b, f)   
                                bidirectional=isBidirectional, #BiLSTM.
                                dropout=dropoutRate,    #Percentage of dropout between non-final layers
                                proj_size=0 #Modifies LSTM to use projections of corresponding size.
                                )            
        

        #Hidden Layers
        self.fc = nn.Linear(hiddenlayerDimensions * (2 if 'bilstm' in modelChosen else 1), outputDimensions)  # 2 for bidirection 

        #Activation Function (always Softmax)
        self.act = nn.Softmax()       

    def forward(self, dataInput, dataLength):

        # # set initial states
        # h0 = torch.zeros(self.num_layers * 2, seq.size(0), self.hidden_size).to(self.device)  # 2 for bidirection
        # c0 = torch.zeros(self.num_layers * 2, seq.size(0), self.hidden_size).to(self.device)
        #   seq = self.embedding(seq)
        # out, _ = self.lstm(seq, (h0, c0))
        #   out, _ = self.lstm(seq)
        # print(out.shape)
        #   out = self.fc(out[:, -1, :])
        #   return F.softmax(out, dim=1)
        # return out
        
        
        
        #Embed
        dataEmbedded = self.embedding(dataInput) #[batchSize, inputLength, embeddingDimensions]

        #Pack
        packed_embedded = nn.utils.rnn.pack_padded_sequence(dataEmbedded, dataLength,batch_first=True)

        #Apply 
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]

        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        #hidden = [batch size, hid dim * num directions]
        dataOutput=self.fc(hidden)

        #Activate
        outputs=self.act(dataOutput)
        
        return outputs

def main(*args):
    
    #Get Config Params
    readConfig(args)
    
    batchSize = 128  
    numEpochs = 10
    bestLoss = float('inf')    
    
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
    
    #Load an iterator
    train_iterator, test_iterator = torchtext.legacy.data.BucketIterator.splits(
        (data_train, data_test), 
        batch_size = batchSize,
        sort_key = lambda x: len(x.Text),
        sort_within_batch=True,
        device = device)    
    
    #Model Construction
    model = questionClassifier(vocabularySize, embeddingDimensions, hiddenlayerDimensions,outputDimensions, layerCount, 
                       isBidirectional = isBidirectional, dropoutRate = dropoutRate)    
    
    #Initialize the pretrained embedding
    if(dataText.vocab.vectors):
        pretrained_embeddings = dataText.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)
    
    #Define the Optim. and Loss metric
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
       
    
def predict(model, sentence):
    nlp = spacy.load('en_core_web_sm')
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence 
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction 
    return prediction.item()     

def trainModel(model, iterator, optimizer, criterion):
        
    epoch_loss, epoch_acc = 0, 0
    
    model.train()  
    
    for batch in iterator:
        
        #Reset grads
        optimizer.zero_grad()   
        
        #Prepare inputs
        text, text_lengths = batch.Text   
        
        #Forward pass (there's an error here)
        predictions = model(text, text_lengths).squeeze()  
        
        #Compute loss/accuracy and backpropogate
        loss = criterion(predictions, batch.Label)        
        acc = binary_accuracy(predictions, batch.Label)   
        loss.backward()       
        optimizer.step()      
        
        #loss and accuracy
        epoch_loss += loss.item()  
        epoch_acc += acc.item()    
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evalModel(model, iterator, criterion):
    
    epoch_loss, epoch_acc = 0, 0

    model.eval()
    
    #deactivates autograd
    with torch.no_grad():
    
        for batch in iterator:
        
            #Prepare inputs
            text, text_lengths = batch.Text
            
            #Forward pass (there's an error here)
            predictions = model(text, text_lengths).squeeze()
            
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