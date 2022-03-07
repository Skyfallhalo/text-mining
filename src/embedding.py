# //////////////////////////////////////////////////////////
#   embedding.py
#   Created on:      01-Mar-2022 14:30:00
#   Original Author: J. Sayce
#   Specification:
#
#   package for several different light-weight embedding methods, 
#   used by question_classifier.py.
#
# //////////////////////////////////////////////////////////

import torch
import torch.nn as nn


def main(vocab_list, config):
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
        trimmed_vocab_list = [""] + vocab_list
        embedding = nn.Embedding(len(trimmed_vocab_list), int(config["Network Structure"]["word_embedding_dim"]))

    embedding.to(device)
    embedding.weight.requires_grad = not config["Model"].getboolean("emb_freeze")

    return embedding, trimmed_vocab_list


# Input: Config directory passed from question_classifier.py
# Task: Populate config values by reading config.ini
# Output: config.
def readConfig(configFile): 
    print("Debug")    


if __name__ == "__main__":
    # main(*sys.argv[1:])
    # sys.exit(0)
    main()
