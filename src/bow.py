#//////////////////////////////////////////////////////////
#   bow.py
#   Created on:      01-Mar-2022 14:30:00
#   Original Author: J. Sayce
#   Specification:
#
#   base file for the 'bow' model, used by question_classifier.py.
#
#//////////////////////////////////////////////////////////

import sys

def main(data, vectorisedWords):

    sentenceVectors = [] # finish list of bag of words sentence vectors

    for sentence in data: 
        
        # vec_bow(s) = 1/length(s) sum for all words in sentence vec(w)

        length = len(sentence) # number of words
        sentenceVector = 0 # for sum of word vectors

        for word in sentence:

            vector = [i for i, v in enumerate(vectorisedWords) if v[0] == word] # find vector for given word in sentence
            sentenceVector += vector 

        sentenceVectors.append(sentenceVector/length)

    return sentenceVectors
    
#Input: Config directory passed from question_classifier.py
#Task: Populate config values by reading config.ini
#Output: config.
def readConfig(configFile): 


    print("Debug")    
    
