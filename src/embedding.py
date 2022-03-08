#//////////////////////////////////////////////////////////
#   embedding.py
#   Created on:      01-Mar-2022 14:30:00
#   Original Author: J. Sayce
#   Specification:
#
#   package for several different light-weight embedding methods, 
#   used by question_classifier.py.
#
#//////////////////////////////////////////////////////////

def main(*args):
    readConfig(args)
    
#Input: Config directory passed from question_classifier.py
#Task: Populate config values by reading config.ini
#Output: config.
def readConfig(configFile): 
    print("Debug")    
    
if __name__ == "__main__":
    main(*sys.argv[1:])
    sys.exit(0)

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import time


class pretrained_embedding:
    
    def __init__(self, corpus, min_count=3,max_count=20,window_size=2,vector_size=10):
        self.corpus = corpus
        self.max_count = max_count
        self.min_count = min_count
        self.window_size = window_size
        self.vector_size  = vector_size
        self.preparation()
        
    def preparation(self):

        """
        create a initial vocabulary with each word's frequency
        Attributes: corpus, 
        """
        # create the vocabulary with each word's frequency
        vocabulary = {}
        for sentence in self.corpus:
            for token in sentence:
                if token not in vocabulary:
                    vocabulary[token] = 1
                else:
                    vocabulary[token] += 1

        # word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
        # idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}



        """
        construct the real dictionay based on frequency.
        remove those most frequency words and those most unfrequency words.
        And add a new category for them: UNK

        Attributes: min_count, max_count, vocabulary_size, word2idx, idx2word

        """
        real_dic = {}
        for i in vocabulary:
            if self.min_count <= vocabulary[i] <= self.max_count:
                real_dic[i] = vocabulary[i]

        l = sorted(real_dic.items(), key = lambda i: i[1],reverse=True) 


        self.vocabulary_size = len(l)+1
        self.word2idx = {w[0]: idx+1 for (idx, w) in enumerate(l)}
        self.idx2word = {idx+1: w[0] for (idx, w) in enumerate(l)}
        self.word2idx['UNK'] = 0
        self.idx2word[0] = 'UNK'


        """
        construct the word pairs with required window size.
        Note that we would ignore all paires with removed words.

        Attributes: corpus, window_size
        """

        idx_pairs = []
        # for each sentence
        for sentence in self.corpus:

            indices = []
            for word in sentence:
                if word in self.word2idx.keys():
                    indices.append(self.word2idx[word])
                else:
                    indices.append(0)

            # for each word, threated as center word
            for center_word_pos in range(len(indices)):
                # for each window position
                for w in range(-self.window_size, self.window_size + 1):
                    context_word_pos = center_word_pos + w
                    # make soure not jump out sentence
                    if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                        continue
                    context_word_idx = indices[context_word_pos]
                    idx_pairs.append((indices[center_word_pos], context_word_idx))
            
            #remove all pairs with 0 values
            self.new_idx_pairs = []
            for i in idx_pairs:
                if i[0] == 0 or i[1] == 0:
                    pass
                else: 
                    self.new_idx_pairs.append(i)
                    
            # numpy array helps to accelerate the training
            self.new_idx_pairs = np.array(self.new_idx_pairs)  



    def get_input_layer(self,word_idx):
        """
        methods for construct input layers.
        For each word it will generate a one-hot encoding format
        
        Attributes: vocabulary_size
        """

        x = torch.zeros(self.vocabulary_size).float()
        x[word_idx] = 1.0
        return x

    def train(self, epochs=5, lr=0.001):
        """
        initialize the weight matrix and other stuff for layers.

        Attributes: vector_size, epochs
        """
        
        embedding_dims = self.vector_size
        W1 = Variable(torch.randn(embedding_dims, self.vocabulary_size).float(), requires_grad=True)
        W2 = Variable(torch.randn(self.vocabulary_size, embedding_dims).float(), requires_grad=True)
        num_epochs = epochs
        learning_rate = lr

        import torch.optim as optim
        # optimizer = optim.SGD([{'params':W1},{'params':W2}],lr = 0.01, momentum = 0.9)
        optimizer = optim.Adam([W1, W2], lr=learning_rate)

        """
        train the model with required epoches for all data pairs.
        Note that there are quite a lot lines for testing time and is commented afterwards.
        
        Attributes: epochs, new_idx_pairs, get_input_layer(),
        """
        start_time = time.time()
        for epo in range(num_epochs):
            loss_val = 0
            for data, target in self.new_idx_pairs:
                x = Variable(self.get_input_layer(data)).float()
                y_true = Variable(torch.from_numpy(np.array([target])).long())
        #         a_time = time.time()
        #         print('create variable time: ', a_time - start_time)

                z1 = torch.matmul(W1, x)
                z2 = torch.matmul(W2, z1)
                log_softmax = F.log_softmax(z2, dim=0)
        #         b_time = time.time()
        #         print('multi time: ' ,b_time - a_time)


                loss = F.nll_loss(log_softmax.view(1,-1), y_true)
                loss_val += loss.item()
                loss.backward()
        #         c_time = time.time()
        #         print('loss backward time: ' ,c_time -b_time)

        #         W1.data -= learning_rate * W1.grad.data
        #         W2.data -= learning_rate * W2.grad.data

        #         W1.grad.data.zero_()
        #         W2.grad.data.zero_()

                optimizer.step()
                optimizer.zero_grad()
        #         d_time = time.time()
        #         print('update time: ' ,d_time-c_time)

        #     if epo % 10 == 0:    
            print(f'Loss at epo {epo}: {loss_val/len(self.new_idx_pairs)}')
            print('time: ', time.time()-start_time)
        self.W2 = W2
    
    def get_embedding(self):
        return self.W2
    
    def length(self, tensor):
        """
        For self use only. should not be called from outside.
        """
        return np.sqrt(sum(np.square(tensor.detach().numpy())))

    def similar(self,word):
        results = []
        aim = self.W2[self.word2idx[word]]
        length_aim = self.length(aim)
        for i in range(len(self.W2)):
            upper = sum((self.W2[i] * aim).detach().numpy())
        #     print('upper:', upper)
            down = length_aim * self.length(self.W2[i])
        #     print('down: ', down)
            if down == 0:
                result = 1
            else:
                result = upper/down
            results.append(-result)

        index = np.argsort(results)
        for i in range(60):
            print(self.idx2word[index[i]])



    