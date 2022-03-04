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



from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def main(*args):
    readConfig(args)
    
    #Example Code
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']     
    
    twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)  
    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42) 
    
    #Vectorising
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    
    #Transformer
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)    
    
    #Classifier (SGD)
    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,max_iter=5, tol=None)\
        .fit(X_train_tfidf, twenty_train.target)
    
    #Display accuracy of predictions 
    docs_test = twenty_test.data
    X_new_counts = count_vect.transform(docs_test)
    new_data = tfidf_transformer.transform(X_new_counts)   
    predicted = clf.predict(new_data)
    
    #Display result of predictions
    for doc, category in zip(docs_test, predicted):
        print('%r \n %s' % (doc, twenty_train.target_names[category]))     
    
    #Display accuracy of predictions
    model_accuracy = round(np.mean(predicted == twenty_test.target)*100, 2) 
    print('Accuracy: {}%'.format(model_accuracy))   
    
    return

    #Make custom predictions
    new_data_raw = ['God is love', 'OpenGL on the GPU is fast']
    X_new_counts = count_vect.transform(twenty_train)
    new_data = tfidf_transformer.transform(X_new_counts)   
    predicted = clf.predict(new_data)
    
    #Display result of predictions
    for doc, category in zip(new_data_raw, predicted):
        print('%r => %s' % (doc, twenty_train.target_names[category]))        
 
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
