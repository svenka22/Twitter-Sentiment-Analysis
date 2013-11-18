import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from DataPreprocessing import DataPreprocessing


# Global Variables #
stopWords=[]
featureList=[]
featureVector=[]
tweets=[]
dataPreprocessing = DataPreprocessing()

class SVMClassifier:

    def getSVMFeatureVectorWithLabels(self,tweets, featureList):        
        labels = []
        feature_vectors = []
         
        for t in tweets:
            
            tweet_words = t[0]
            tweet_opinion = t[1]
            feature_vector = []
            
            for feature_word in featureList:
                # set map[word] to 1 if word exists
                if feature_word in tweet_words:
                    feature_vector.append(1)
                else:
                    feature_vector.append(0)
            if(tweet_opinion=="Cessation"):
                labels.append(1)
            elif(tweet_opinion=="No Cessation"):
                labels.append(0)
                
            feature_vectors.append(feature_vector)
        return feature_vectors,labels
    # end
    
    
    #Main function
    def SVMbuildClassifier(self):
        global stopWords
        global featureList
        global tweets
        
        stopWords = nltk.corpus.stopwords.words('english')
        print "Stop words retrieved"
        
        #Read the tweets one by one and process it
        fp = open('../data/smokingtweets.txt', 'r', )
        line = fp.readline()
        i = 1
        while line:
            linesplit = line.split('|@~')
            tweet = linesplit[0]
            #print "tweet: ", tweet
            sentiment = linesplit[1].rstrip()
            #print "sentiment: ", sentiment
            processedTweet = dataPreprocessing.processTweet(tweet)
            #print "processedTweet:",processedTweet
            featureVector = dataPreprocessing.getFeatureVector(processedTweet)
            #print "featureVector:",featureVector
            tweets.append((featureVector, sentiment));
            featureList = dataPreprocessing.union(featureList, featureVector)
            #print ""
            line = fp.readline()
            i = i + 1
        fp.close()
        print "Data Preprocessing Completed..."
        print "Features Extracted..."
        
        # Run SVM Classifier
        SVMClassifier = svm.SVC(kernel='linear')
        pred_label = []
        
        print "Classifying.."
        num_folds = 10
        Accuracy = 0
        subset_size = len(tweets)/num_folds
        for i in range(num_folds):
            testing_this_round = tweets[i*subset_size:][:subset_size]
            training_this_round = tweets[:i*subset_size] + tweets[(i+1)*subset_size:]
            feature_vectors_train,training_labels = self.getSVMFeatureVectorWithLabels(training_this_round, featureList)
            feature_vectors_test,actual_label = self.getSVMFeatureVectorWithLabels(testing_this_round, featureList)
            
            pred_label = SVMClassifier.fit(feature_vectors_train, training_labels).predict(feature_vectors_test)
            #pred_label_list.extend(pred_label)    
            Accuracy = Accuracy + accuracy_score(actual_label, pred_label)        
        print "Accuracy: ",float(float(Accuracy)/float(num_folds))
    
