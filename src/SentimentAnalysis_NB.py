import itertools
import re
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.metrics import accuracy_score, confusion_matrix

# package for data preprocessing
from DataPreprocessing import DataPreprocessing

# Global Variables #
stopWords=[]
featureList=[]
featureVector=[]
tweets=[]
dataPreprocessing = DataPreprocessing()

class NBClassifier:

    def crossValidation(self, training_set):
        num_folds=10
        Accuracy=0
        subset_size = len(training_set)/num_folds
        for i in range(num_folds):
            testing_this_round = training_set[i*subset_size:][:subset_size]
            training_this_round = training_set[:i*subset_size] + training_set[(i+1)*subset_size:]
            classifier = nltk.classify.NaiveBayesClassifier.train(training_this_round)
            Accuracy=Accuracy+nltk.classify.accuracy(classifier, testing_this_round)
        #Total Accuracy    
        Accuracy=float(float(Accuracy)/float(num_folds))
        return Accuracy

    # This method uses the training set to train the NBClassifer
    # and constructs the classifier
    # The test tweets are read and tested against the trained classifier
    # The accuracy score, precision and recall is calculated
    def plainVaidation(self, training_set):
        
        # Train the classifier
        NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
        
        #Reading the test data
        tp = open('../data/testtweets.txt', 'r')
        tLine = tp.readline()
        
        actual_class = []
        pred_class = []
        while tLine:
            lines = tLine.rstrip().split('|@~')
            tweet = lines[0]
            sentiment = lines[1]
            #Predict the class using NB Classifier and append it to the pred_class list
            pred_class.append(NBClassifier.classify(dataPreprocessing.extract_features(\
                    dataPreprocessing.getFeatureVector(dataPreprocessing.processTweet(tweet)))))
            actual_class.append(sentiment)
            tLine = tp.readline()
        # end loop
        
        # Confusion Matrix Construction
        cm = confusion_matrix(actual_class, pred_class)
        print cm
        accuracyScoreValue = accuracy_score(actual_class, pred_class)
        precision = nltk.metrics.precision(set(actual_class),set(pred_class));
        recall = nltk.metrics.recall(set(actual_class),set(pred_class));
        f_score = nltk.metrics.f_measure(set(actual_class),set(pred_class), 0.5)
        # returns the accuracy, 
        # actualClass from the test data,
        # predicted class for test tweets from the clasifer
        return accuracyScoreValue,\
            actual_class, pred_class,precision,recall,f_score
                
    
   
    #Main function
    def NBbuildClassifier(self):
        global stopWords
        global featureList
        global tweets
        
        # retrieve the stop words in english using the nltk package
        stopWords = nltk.corpus.stopwords.words('english')
        
        #Read the tweets one by one and process it
        fp = open('../data/smokingtweets.txt', 'r', )
        line = fp.readline()
        
        # pre process the tweet and vectorize the tweet
        # create a list of dict of the form [feature_vector, sentiment]
        while line:
            linesplit = line.split('|@~')
            tweet = linesplit[0]
            sentiment = linesplit[1].rstrip()
            processedTweet = dataPreprocessing.processTweet(tweet)
            featureVector = dataPreprocessing.getFeatureVector(processedTweet)
            tweets.append((featureVector, sentiment));
            line = fp.readline()
        fp.close()
        
        
        training_set = nltk.classify.util.apply_features(dataPreprocessing.extract_features, tweets)
        Accuracy,actualClass,predictClass,precision,recall,f_score = self.plainVaidation(training_set)
        print "Accuracy using Naive Bayesian Classification:",Accuracy
        print "Precision using Naive Bayesian Classification:",precision
        print "Recall using Naive Bayesian Classification:",recall
        print "F-Score using Naive Bayesian Classification:",f_score
            
