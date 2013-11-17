import re
import nltk
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

# Global Variables #
stopWords=[]
featureList=[]
featureVector=[]
tweets=[]

#start process_tweet
def processTweet(tweet):
    # process the tweets
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end
 
#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end
 
#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')
 
    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end
 
#start getfeatureVector
def getFeatureVector(tweet):
    #global featureVector
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#end


def getSVMFeatureVectorWithLabels(tweets, featureList):
    
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

def union(a, b):
    return list(set(a) | set(b))

#start extract_features
def extract_features(tweet, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(tweet)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(tweet, bigrams)])
#end

# #start extract_features
# def extract_features(tweet):
#     tweet_words = set(tweet)
#     features = {}
#     for word in featureList:
#         features['contains(%s)' % word] = (word in tweet_words)
#     return features
# #end

#Main function
def buildClassifier():
    global stopWords
    global featureList
    global tweets
    
    #stopWords = getStopWordList('data/stopwords.txt')
    stopWords=nltk.corpus.stopwords.words('english')
    
    #Read the tweets one by one and process it
    fp = open('data/smokingtweets.txt', 'r', )
    line = fp.readline()
    
    while line:
        linesplit = line.split('|@~')
        tweet = linesplit[0]
        sentiment = linesplit[1].rstrip()
        processedTweet = processTweet(tweet)
        featureVector = getFeatureVector(processedTweet)
        tweets.append((featureVector, sentiment));
        #featureList = union(featureList, featureVector)
        line = fp.readline()
    fp.close()
    
    training_set = nltk.classify.util.apply_features(extract_features, tweets)
    # Train the classifier
    #NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
    
    #Reading the test data
#     tp = open('data/testdata.txt', 'r')
#     tLine = tp.readline()
#     
#     actual_class = []
#     pred_class = []
#     while tLine:
#         lines = tLine.rstrip().split('|@~')
#         tweet = lines[0]
#         sentiment = lines[1]
#         #Predict the class using NB Classifier and append it to the pred_class list
#         pred_class.append(NBClassifier.classify(extract_features(getFeatureVector(processTweet(tweet)))))
#         actual_class.append(sentiment)
#         tLine = tp.readline()
#     # end loop
    
    num_folds=10
    Accuracy=0
    subset_size = len(training_set)/num_folds
    for i in range(num_folds):
        testing_this_round = training_set[i*subset_size:][:subset_size]
        training_this_round = training_set[:i*subset_size] + training_set[(i+1)*subset_size:]
        classifier = nltk.classify.NaiveBayesClassifier.train(training_this_round)
        Accuracy=Accuracy+nltk.classify.accuracy(classifier, testing_this_round)
        
    print "Mean Accuracy for DT",float(float(Accuracy)/float(num_folds))
    
#     cm = confusion_matrix(actual_class, pred_class)
#     print cm
#     acc = accuracy_score(actual_class, pred_class)
#     print "Accuracy using Naive Bayesian Classification:",acc

# Call the main method
if __name__ == '__main__':  buildClassifier() 