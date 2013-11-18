import itertools
import re

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.metrics import accuracy_score, confusion_matrix

from DataPreprocessing import DataPreprocessing
# Global Variables #
stopWords=[]
featureList=[]
featureVector=[]
tweets=[]
dataPreprocessing = DataPreprocessing()

class NBClassifier:

    #Main function
    def NBbuildClassifier(self):
        global stopWords
        global featureList
        global tweets
        
        stopWords = nltk.corpus.stopwords.words('english')
        
        #Read the tweets one by one and process it
        fp = open('../data/smokingtweets.txt', 'r', )
        line = fp.readline()
        
        while line:
            linesplit = line.split('|@~')
            tweet = linesplit[0]
            sentiment = linesplit[1].rstrip()
            processedTweet = dataPreprocessing.processTweet(tweet)
            featureVector = dataPreprocessing.getFeatureVector(processedTweet)
            tweets.append((featureVector, sentiment));
            #featureList = union(featureList, featureVector)
            line = fp.readline()
        fp.close()
        
        training_set = nltk.classify.util.apply_features(dataPreprocessing.extract_features, tweets)
    
        num_folds=10
        Accuracy=0
        subset_size = len(training_set)/num_folds
        for i in range(num_folds):
            testing_this_round = training_set[i*subset_size:][:subset_size]
            training_this_round = training_set[:i*subset_size] + training_set[(i+1)*subset_size:]
            classifier = nltk.classify.NaiveBayesClassifier.train(training_this_round)
            Accuracy=Accuracy+nltk.classify.accuracy(classifier, testing_this_round)
            
        print "Mean Accuracy for DT",float(float(Accuracy)/float(num_folds))
        
#         cm = confusion_matrix(actual_class, pred_class)
#         print cm
#         acc = accuracy_score(actual_class, pred_class)
#         print "Accuracy using Naive Bayesian Classification:",acc
    
