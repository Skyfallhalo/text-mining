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

import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.datasets._base import Bunch
from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def main(*args):
    readConfig(args)
    
    
def sklearnMethod():
    
    train_file, train_targets = getFromFile('../data/train.txt')
    test_file, test_targets = getFromFile('../data/test.txt')
    
    #Generate universal category list, convert targets into indexes
    categories = list(set(train_targets + test_targets))
    train_targets = [categories.index(i) for i in train_targets]
    test_targets = [categories.index(i) for i in test_targets]
    
    #Data: Questions
    #Target_Names: Classifications
    #Target: Indexes of classifications
    train_data = Bunch(data=train_file, target=train_targets, target_names=categories)
    test_data = Bunch(data=test_file, target=test_targets, target_names=categories)
    
    #Vectorising
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_data.data)
    
    #Transformer
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)    
    
    #Classifier (SGD)
    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,max_iter=5, tol=None)\
        .fit(X_train_tfidf, train_data.target)
    
    #Display accuracy of predictions 
    docs_test = test_data.data
    X_new_counts = count_vect.transform(docs_test)
    new_data = tfidf_transformer.transform(X_new_counts)   
    predicted = clf.predict(new_data)
    
    #Display result of predictions
    for doc, category, actual in zip(docs_test, predicted, test_data.target):
        print('%r %s \n %s (%s) \n' % (doc, "Correct" if category == actual else "Incorrect", test_data.target_names[category], test_data.target_names[actual]))     
    
    #Display accuracy of predictions
    score_accuracy = accuracy_score(test_data.target, predicted)
    f1_accuracy = f1_score(test_data.target, predicted, average='micro')
    print('Accuracy Score: {}%. F1 Accuracy: {}%.'.format(score_accuracy*100, f1_accuracy*100))   
    
    sample_question = ['What are shadows made of?']
    X_new_counts = count_vect.transform(sample_question)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    predicted = clf.predict(X_new_tfidf)
    
    for doc, category in zip(sample_question, predicted):
        print('%r => %s' % (doc, train_data.target_names[category]))    
     
 
def getFromFile(directory):
    data = []
    targets = []
    with open(directory) as my_file:
        for line in my_file:
            delimline = line.split(" ", 1)
            data.append(delimline[1])  
            targets.append(delimline[0])
    return data, targets
 
def predictData(dataset, new_data_raw, new_data, clf):
    
    #Make predictions on test data
    new_predicted = clf.predict(new_data)
    
    #Display result of predictions
    for doc, category in zip(new_data_raw, new_predicted):
        print('%r => %s' % (doc, dataset.target_names[category]))      
      
    
    
#Input: Config directory passed from question_classifier.py
#Task: Populate config values by reading config.ini
#Output: config.
def readConfig(configFile): 
    print("Debug")    
    
if __name__ == "__main__":
    main(*sys.argv[1:])
    sys.exit(0)
