import time
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
from sklearn import svm
from sklearn.metrics import accuracy_score
from DataPreprocessing import DataPreprocessing


start_time=time.time()

# Global Variables #
stopWords=[]
featureList=[]
featureVector=[]
tweets=[]
dataPreprocessing = DataPreprocessing()

class SVMClassifier:
    
    def crossValidation(self, SVMClassifier):
        global featureList
        global tweets
        
        num_folds=5
        Accuracy=0
        subset_size = len(tweets)/num_folds
        for i in range(num_folds):
            pred_label_list = []
            testing_this_round = tweets[i*subset_size:][:subset_size]
            training_this_round = tweets[:i*subset_size] + tweets[(i+1)*subset_size:]
            
            for training_tweet in training_this_round:
                featureList = dataPreprocessing.union(featureList, training_tweet[0])
                #bigrams = dataPreprocessing.get_bigrams(training_tweet[0])
                #featureList = dataPreprocessing.union(featureList, bigrams)
            #end-loop
            
            feature_vectors_train,training_labels = self.getSVMFeatureVectorWithLabels(training_this_round, featureList)
            feature_vectors_test,actual_label_list = self.getSVMFeatureVectorWithLabels(testing_this_round, featureList)
            pred_label = SVMClassifier.fit(feature_vectors_train, training_labels).predict(feature_vectors_test)
            pred_label_list.extend(pred_label)
            Accuracy = Accuracy+accuracy_score(actual_label_list, pred_label_list)
            print Accuracy
            
        Accuracy=float(float(Accuracy)/float(num_folds))
        return Accuracy

        
    def plainValidation(self, SVMClassifier):
        
        testTweets = []
        pred_label_list = []
        global featureList
        global tweets
        
        #Building featureList
        for tweet in tweets:
                #bigrams = dataPreprocessing.get_bigrams(tweet[0])
                #featureList = dataPreprocessing.union(featureList, bigrams)
                featureList = dataPreprocessing.union(featureList, tweet[0])
        
        #print "featureList:",featureList
        # Train the SVM Classifier
        feature_vectors_train,training_labels = self.getSVMFeatureVectorWithLabels(tweets, featureList)
        print "SVM Classifier trained"
        
        #Read the test tweet
        tp = open('../data/testtweets.txt', 'r')
        tLine = tp.readline()
        while tLine:
            lines = tLine.rstrip().split('|@~')
            testTweet = lines[0]
            sentiment = lines[1]
            processedtestTweet = dataPreprocessing.processTweet(testTweet)
            featureVector = dataPreprocessing.getFeatureVector(processedtestTweet)
            #featureVector = featureVector + dataPreprocessing.get_bigrams(featureVector)
            testTweets.append((featureVector, sentiment))
            tLine = tp.readline()
        # end loop
        
        # Extract features from Test data
        feature_vectors_test,actual_label_list = self.getSVMFeatureVectorWithLabels(testTweets, featureList)
        
        pred_label = SVMClassifier.fit(feature_vectors_train, training_labels).predict(feature_vectors_test)
        pred_label_list.extend(pred_label)
        #cm = confusion_matrix(actual_label_list, pred_label_list)
        #print cm
        acc = accuracy_score(actual_label_list, pred_label_list)
        return acc
    
    
    def getSVMFeatureVectorWithLabels(self, tweets, featureList):        
        labels = []
        feature_vectors = []
        tweet_words = []
        
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
        #global featureList
        global tweets
        
        #Read the tweets one by one and process it
        fp = open('../data/smokingtweets.txt', 'r', )
        line = fp.readline()
        while line:
            linesplit = line.split('|@~')
            tweet = linesplit[0]
            sentiment = linesplit[1].rstrip()
            processedTweet = dataPreprocessing.processTweet(tweet)
            featureVector = dataPreprocessing.getFeatureVector(processedTweet)
            #bigrams = dataPreprocessing.get_bigrams(featureVector)
            #featureVector = dataPreprocessing.union(featureVector, bigrams)
            tweets.append((featureVector, sentiment));
            line = fp.readline()
        fp.close()
        print "Data Preprocessing Completed..."
        
        # Run SVM Classifier
        SVMClassifier = svm.SVC(kernel='linear')
        Accuracy=self.crossValidation(SVMClassifier)   
        #Accuracy=self.plainValidation(SVMClassifier)  
        
        print "Accuracy using SVM:",Accuracy
        print "Time:",str(time.time()-start_time)
