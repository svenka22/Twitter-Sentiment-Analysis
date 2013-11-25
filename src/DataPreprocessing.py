import en
import itertools
import re
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import wordnet
from nltk.metrics import BigramAssocMeasures
from nltk.stem.wordnet import WordNetLemmatizer



class DataPreprocessing:
    
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
        featureVector.append(w.lower())
        return featureVector
       
    #start getfeatureVector
    def getFeatureVector(self, tweet):
        
        stopWords=[]
        #retrieve the stop words in english using the nltk package
        stopWords = nltk.corpus.stopwords.words('english')
        stopWords.append("URL")
        stopWords.append("AT_USER")
        
        #global featureVector
        featureVector = []
        emoticons = [':-)',':)','(-:','(:',';)',';-)',')-:','):',':-(',':(',':-P','=P',':P',':-D',':\'(',':-/',':S','=D','<3',':D', ':|',':-|']
        #split tweet into words
        #print tweet
        words = tweet.split()
        for w in words:        #replace two or more with two occurrences
            if w in emoticons:
                featureVector.append(w)
            w = self.replaceTwoOrMore(w)
            #strip punctuation
            w = w.strip('\'"?,.')
            #check if the word stats with an alphabet
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
            #ignore if it is a stop word
            if(w in stopWords or val is None):
                continue
            else:
                #self.lematizer(w, featureVector)
                featureVector.append(w.lower())
        return featureVector
    #end
       
    def union(self,a, b):
        return list(set(a) | set(b))
    
    #start extract_features
    def get_bigrams(self, tweet, score_fn=BigramAssocMeasures.chi_sq, n=200):
            bigramslist = []
            bigram_finder = BigramCollocationFinder.from_words(tweet)
            bigrams = bigram_finder.nbest(score_fn, n)
            for bigram in bigrams:
                bigramslist.append(' '.join(str(i) for i in bigram))
            return bigramslist #This is list e.g. ['you dude', 'Hi How', 'How are', 'are you']
    #end 
    
    #removes duplicates
    #input: list
    #output: list
    def removeDup(self,l):
        return list(set(l))
    
    #Get the synonyms for the given string input
    def getSynonyms(self,word,featureList):
        list_of_list_synonyms = en.noun.senses(word)        #not an error
        if len(list_of_list_synonyms) != 0:
            merged = list(itertools.chain(*list_of_list_synonyms))
            featureList = self.removeDup(merged) + featureList
        return featureList
            
    