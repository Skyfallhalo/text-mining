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


import torch
import torch.nn as nn
#import sys



def main(config,corpus=None):
    # readConfig(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trimmed_vocab_list = []
    if config["Model"].getboolean("pre_emb"):
        vec_list = []
        vec_size = 0
        with open(config["Using pre-trained Embeddings"]["path_pre_emb"]) as f:
            for line in f:
                [glove_word, vec_str] = line.split("\t", 1)
                glove_word = glove_word.strip().lower()
                idx = -1
                for i in range(len(vocab_list)):
                    if glove_word == vocab_list[i]:
                        idx = i
                        break
                # if idx != -1 or glove_word == "#unk#":
                if idx != -1:
                    trimmed_vocab_list.append(glove_word)
                    vec = list(map(lambda x: float(x), vec_str.strip().split(" ")))
                    if len(vec) > vec_size:
                        vec_size = len(vec)
                    vec_list.append(vec)
                    del vocab_list[idx]
        trimmed_vocab_list.insert(0, "")
        vec_list.insert(0, [0.0] * vec_size)

        # for e in vocab_list:
        #     print(e)
        # print(len(vocab_list))

        weight = torch.FloatTensor(vec_list)
        # embedding = nn.Embedding.from_pretrained(weight, freeze=freeze)
        embedding = nn.Embedding.from_pretrained(weight)
    else:
       # trimmed_vocab_list = [""] + vocab_list
       # embedding = nn.Embedding(len(trimmed_vocab_list), int(config["Network Structure"]["word_embedding_dim"]))

        if corpus is not None:
            vocabulary = {}
            for sentence in corpus:
                for token in sentence:
                    if token not in vocabulary:
                        vocabulary[token] = 1
                    else:
                        vocabulary[token] += 1


            trimmed_vocab_list = []
            for i in vocabulary:
                if min_count <= vocabulary[i]:
                    trimmed_vocab_list.append(i)

            trimmed_vocab_list.append('UNK')
            trimmed_vocab_list.insert(0,'')

            vocabulary_size = len(trimmed_vocab_list)
            embedding = nn.Embedding(vocabulary_size, int(config["Network Structure"]["word_embedding_dim"]))
        
        else:
            sys.exit('No corpus to generate vocabulary and random embedding')
        

    embedding.to(device)
    embedding.weight.requires_grad = not config["Model"].getboolean("emb_freeze")

    return embedding, trimmed_vocab_list





#def main(config,corpus=None):
#    """
#    For use this module:
#    1.add import sys
#    2.add path_pre_emb_vocab in config
#    3.
#    """
#    # readConfig(args)
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#    vocab_list = []
#    if config["Model"].getboolean("pre_emb"):
#        with open(config["Using pre-trained Embeddings"]["path_pre_emb"]) as f:
#            vocab_list = f.read().split(',')
#
#        while '' in vocab_list:
#            vocab_list.remove('')
#        vocab_list.insert(0,'')
#
#        #load embedding vectors
#        with open(config["Using pre-trained Embeddings"]["path_pre_emb_vocab"]) as my_file:
#            content_list = my_file.read().split(';')
#
#
#        vectors = []
#        for vector in content_list:
#            vec = []
#            elements = vector.split(' ')
#            for i in elements:
#                if '[' in i:
#                    i = i.replace('[', '')
#                if ']' in i:
#                    i = i.replace(']', '')
#                if '\n' in i:
#                    i = i.replace('\n', '')
#                if i != '':
#                    vec.append(float(i))
#            vectors.append(vec)
#
#        vectors.remove([])
#        vecs = np.array(vectors)
#        a = np.zeros((1,vecs.shape[1]))
#        vecs2 = np.concatenate((a,vecs))
#
#        weights = torch.FloatTensor(vecs2)
#        embedding = nn.Embedding.from_pretrained(weights,freeze=freeze)
#
#
#
#    else:
#        if corpus is not None:
#            vocabulary = {}
#            for sentence in corpus:
#                for token in sentence:
#                    if token not in vocabulary:
#                        vocabulary[token] = 1
#                    else:
#                        vocabulary[token] += 1
#
#
#            vocab_list = []
#            for i in vocabulary:
#                if min_count <= vocabulary[i]:
#                    vocab_list.append(i)
#
#            vocab_list.append('UNK')
#            vocab_list.insert(0,'')
#
#            vocabulary_size = len(vocab_list)
#            embedding = nn.Embedding(vocabulary_size, vector_size)
#
#        else:
#            sys.exit('No corpus to generate vocabulary and random embedding')
#
#
#    embedding.to(device)
#    embedding.weight.requires_grad = not config["Model"].getboolean("emb_freeze")
#
#    return embedding, vocab_list


# Input: Config directory passed from question_classifier.py
# Task: Populate config values by reading config.ini
# Output: config.
def readConfig(configFile):
    print("Debug")    


if __name__ == "__main__":
    # main(*sys.argv[1:])
    # sys.exit(0)
    main()





