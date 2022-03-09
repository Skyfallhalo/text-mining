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

#def main(*args):
#    readConfig(args)
    
#Input: Config directory passed from question_classifier.py
#Task: Populate config values by reading config.ini
#Output: config.
#def readConfig(configFile): 
 #   print("Debug")    
    
#if __name__ == "__main__":
#    main(*sys.argv[1:])
#    sys.exit(0)

import numpy as np
import torch

def pretrained_embedding(vocab_fp, emb_fp):
    """
    return the pretrained embeddings
    
    Attributes:
      vocab_fp: file path for vocabulary file (.txt)
      emb_fp: file path for vector file (.txt)
      
    Returns:
      vocab_list: a list that contains all words (including 'UNK' for words not in it)
      emb: the nn.embedding type that contains embedding informations for all vocab 
    """
    
    #load the vocab
    with open(vocab_fp, "r") as my_file:
        vocab_list = my_file.read().split(',')

    vocab_list.remove('')


    #load embedding vectors
    with open(emb_fp, "r") as my_file:
        content_list = my_file.read().split(';')


    vectors = []
    for vector in content_list:
        vec = []
        elements = vector.split(' ')
        for i in elements:
            if '[' in i:
                i = i.replace('[', '')
            if ']' in i:
                i = i.replace(']', '')
            if '\n' in i:
                i = i.replace('\n', '')
            if i != '':
                vec.append(float(i))

        vectors.append(vec)

    vectors.remove([])  

    vec = np.array(vectors)
    weights = torch.FloatTensor(vec)
    emb = nn.Embedding.from_pretrained(weights)
        
    return vocab_list, emb
    
def random_embedding(corpus, min_count=3, vector_size=200):
    """
    return random initialized embeddings.
    
    Attributes:
      corpus: list of lists, and each list contains words from each sentences.
      min_count: the min count for words to appear in the vocabulary
      vector_size: embedding dimensions
      
      
    Returns:
      real_vocab: a list that contains all words (including 'UNK' for words not in it)
      emb: the nn.embedding type that contains embedding informations for all vocab 
    """
    vocabulary = {}
    for sentence in corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary[token] = 1
            else:
                vocabulary[token] += 1


    real_vocab = []
    for i in vocabulary:
        if min_count <= vocabulary[i]:
            real_vocab.append(i)

    real_vocab.append('UNK')

    vocabulary_size = len(real_vocab)
    emb = nn.Embedding(vocabulary_size, vector_size)

    return real_vocab, emb





