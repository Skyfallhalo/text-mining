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

import torch
import torch.nn as nn

def main(vocab_list, config):
    """
    return embedding result and vocabulary.
    
    For pre0trained embeddings:
     Config file:
       pre_emb : true
       path_pre_emb: a path to the embedding txt file
       emb_freeze: true or false
     Attributes:
       vocab_list: a list of all words from the training data.
    
    For random initialized embeddings:
      Config file:
        pre_emb : false
        word_embedding_dim: choose a value between 100 ~ 300
      Attributes:
        vocab_list: a list of words from training data.
    
    Returns:
      embedding: a nn.embedding variable
      trimmed_vocab_list: a trimmed final vocab list
    """

    if config["Model"].getboolean("emb_freeze") and not config["Model"].getboolean("pre_emb"):
        raise Exception("The freezing randomly initialised vectors are not supported.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trimmed_vocab_list = []
    # Case when pretrained vectors are used
    if config["Model"].getboolean("pre_emb"):
        vec_list = []
        vec_size = 0
        with open(config["Using pre-trained Embeddings"]["path_pre_emb"]) as f:
            for line in f:
                # Split word and vector string
                [glove_word, vec_str] = line.split("\t", 1)
                glove_word = glove_word.strip().lower()

                idx = -1
                # Get index of a glove word in the vocabulary list
                for i in range(len(vocab_list)):
                    if glove_word == vocab_list[i]:
                        idx = i
                        break
                # When a word is in the vocabulary list
                if idx != -1:
                    trimmed_vocab_list.append(glove_word)
                    # Get vector list
                    vec = list(map(lambda x: float(x), vec_str.strip().split(" ")))
                    if len(vec) > vec_size:
                        vec_size = len(vec)
                    vec_list.append(vec)
                    # Delete the found word in original vocabulary list for faster retrieval later on
                    del vocab_list[idx]

        # Add vector for padding index represented by an empty string
        trimmed_vocab_list.insert(0, "")
        vec_list.insert(0, [0.0] * vec_size)

        weight = torch.FloatTensor(vec_list)
        embedding = nn.Embedding.from_pretrained(weight)
    # Case when vectors are initialised randomly (vector dimension is required)
    else:
        # Add vector for padding index represented by an empty string
        vocab_list.insert(0, '')
        trimmed_vocab_list = vocab_list

        vocabulary_size = len(trimmed_vocab_list)
        embedding = nn.Embedding(vocabulary_size, int(config["Network Structure"]["word_embedding_dim"]))

    embedding.to(device)
    # Set whether the embeddings should be fine-tuned during data training
    embedding.weight.requires_grad = not config["Model"].getboolean("emb_freeze")

    return embedding, trimmed_vocab_list





