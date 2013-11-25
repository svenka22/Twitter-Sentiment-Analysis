import time
import nltk
from sklearn.metrics import accuracy_score, confusion_matrix

from DataPreprocessing import DataPreprocessing
from Metrics import Metrics


start_time=time.time()

# package for data preprocessing

# Global Variables #
stopWords=[]
featureList=[]
featureVector=[]
tweets=[]

dataPreprocessing = DataPreprocessing()
metrics = Metrics()

class NBClassifier:
    
    #start extract_features
    def extract_features(self, tweet):
        tweet_words = set(tweet)
        features = {}
        for word in featureList:
            features['contains(%s)' % word] = (word in tweet_words)
        return features
    #end
    
    def crossValidation(self, tweets):
        #tweets is a list e.g.: [(['electronic', 'cigarette', 'risk'], 'No Cessation'), (['word'],class) ]
        global featureList
        num_folds=5
        Accuracy=0
        subset_size = len(tweets)/num_folds
        for i in range(num_folds):
            testing_this_round = tweets[i*subset_size:][:subset_size]
            training_this_round = tweets[:i*subset_size] + tweets[(i+1)*subset_size:]
            
            for training_tweet in training_this_round:
                featureList = dataPreprocessing.union(featureList, training_tweet[0])
                bigrams = dataPreprocessing.get_bigrams(training_tweet[0])
                featureList = dataPreprocessing.union(featureList, bigrams)
            #end-loop
            
            training_set = nltk.classify.util.apply_features(self.extract_features, training_this_round)
            classifier = nltk.classify.NaiveBayesClassifier.train(training_set)
            
            test_set = nltk.classify.util.apply_features(self.extract_features, testing_this_round)
            Accuracy=Accuracy+nltk.classify.accuracy(classifier, test_set)
            print Accuracy
        Accuracy=float(float(Accuracy)/float(num_folds))
        return Accuracy

    def plainValidation(self, tweets):
        #print tweets
        #tweets is a list e.g.: [(['electronic', 'cigarette', 'risk'], 'No Cessation'), (['word'],class) ]
        global featureList
        actual_class = []
        pred_class = []
        
        #Building featureList
        for tweet in tweets:
                #bigrams = dataPreprocessing.get_bigrams(training_tweet[0])
                #for each word, compute the synonyms and add to feature list
                for word in tweet[0]:
                    featureList = dataPreprocessing.getSynonyms(word, featureList)
                # remove duplicate words
                featureList = dataPreprocessing.removeDup(featureList)
                # add the tweet words
                featureList = dataPreprocessing.union(featureList, tweet[0])
                
        print #feature list prepared..
        # Train the classifier
        training_set = nltk.classify.util.apply_features(self.extract_features, tweets)
        NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
        print "Classifier trained"
                
        print "Reading test data"
        #Reading the test data
        tp = open('../data/testtweets.txt', 'r')
        tLine = tp.readline()
        
        while tLine:
            lines = tLine.rstrip().split('|@~')
            tweet = lines[0]
            sentiment = lines[1]
            processedTweet = dataPreprocessing.processTweet(tweet)
            testFeatureVector = dataPreprocessing.getFeatureVector(processedTweet)
            #bigrams = dataPreprocessing.get_bigrams(testFeatureVector)
            #testFeatureVector = dataPreprocessing.union(bigrams, testFeatureVector)
            pred_class.append(NBClassifier.classify(self.extract_features(testFeatureVector)))
            #print "pred_class:",pred_class
            actual_class.append(sentiment)
            tLine = tp.readline()
        # end loop
        print #done classifying Calculating metrics
        cm = confusion_matrix(actual_class, pred_class)
        print cm
        acc = accuracy_score(actual_class, pred_class)
        #Print the metrics for the classifier result
        metrics.calculateClassifierMetrics(actual_class, pred_class)
        return acc
    
   
    #Main function
    def NBbuildClassifier(self):
        global stopWords
        global featureList
        global tweets
      
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
            bigrams = dataPreprocessing.get_bigrams(featureVector)
            featureVector = dataPreprocessing.union(featureVector, bigrams)
            tweets.append((featureVector, sentiment));
            line = fp.readline()
        fp.close()
        print "Preprocessing done"
        
        Accuracy = self.plainValidation(tweets)
        #Accuracy = self.crossValidation(tweets)
        print "Accuracy using Naive Bayesian Classification:",Accuracy
        print "Time:",str(time.time()-start_time)
