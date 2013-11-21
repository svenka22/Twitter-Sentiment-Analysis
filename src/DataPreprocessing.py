import re
import nltk
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

class DataPreprocessing:
    
    stopWords=[]
    
    #start process_tweet
    def processTweet(self,tweet):
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
    def replaceTwoOrMore(self,s):
        #look for 2 or more repetitions of character and replace with the character itself
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        return pattern.sub(r"\1\1", s)
    #end
    
    def lematizer(self, w, featureVector):
        
        lmtzr = WordNetLemmatizer()
        tokens=nltk.word_tokenize(w)
        grammer_tuple = nltk.pos_tag(tokens)
        #print w,"-->", grammer_tuple.pop()[1]
        grammer =  grammer_tuple.pop()[1]
        #print "Before:", w
        if grammer.startswith('J'):
            w = lmtzr.lemmatize(w, wordnet.ADJ)
        elif grammer.startswith('V'):
            w = lmtzr.lemmatize(w, wordnet.VERB)
        elif grammer.startswith('N'):
            w = lmtzr.lemmatize(w, wordnet.NOUN)
        elif grammer.startswith('R'):
            w = lmtzr.lemmatize(w, wordnet.ADV)
        else:
            w = lmtzr.lemmatize(w, wordnet.NOUN)
        #print "After:", w, "\n"
        featureVector.append(w.lower())
        return featureVector
       
    #start getfeatureVector
    def getFeatureVector(self, tweet):
        #global featureVector
        featureVector = []
        #split tweet into words
        words = tweet.split()
        for w in words:        #replace two or more with two occurrences
            w = self.replaceTwoOrMore(w)
            #strip punctuation
            w = w.strip('\'"?,.')
            #check if the word stats with an alphabet
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
            #ignore if it is a stop word
            if(w in self.stopWords or val is None):
                continue
            else:
                #self.lematizer(w, featureVector)
                featureVector.append(w.lower())
        return featureVector
    #end
       
    def union(self,a, b):
        return list(set(a) | set(b))
    
    
    #start extract_features
    def extract_features(self,tweet, score_fn=BigramAssocMeasures.chi_sq, n=200):
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
#end