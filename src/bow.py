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

    sentenceVectors = []

    for sentence in data:

        length = len(sentence)
        sentenceVector = 0

        for word in sentence:

            vector = [i for i, v in enumerate(vectorisedWords) if v[0] == word]
            sentenceVector += vector

        sentenceVectors.append(sentenceVector/length)

    return sentenceVectors
    
#Input: Config directory passed from question_classifier.py
#Task: Populate config values by reading config.ini
#Output: config.
def readConfig(configFile): 


    print("Debug")    
    
