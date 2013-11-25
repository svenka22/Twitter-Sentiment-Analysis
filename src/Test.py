import re
import nltk
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

#start extract_features
def get_bigrams1(tweet, score_fn=BigramAssocMeasures.chi_sq, n=200):
        bigramslist = []
        bigram_finder = BigramCollocationFinder.from_words(tweet)
        bigrams = bigram_finder.nbest(score_fn, n)
        for bigram in bigrams:
            bigramslist.append(' '.join(str(i) for i in bigram))
        print bigramslist
#end

if __name__ == '__main__':  get_bigrams1(["Hi", "How", "are", ":)", "dude", "morning"]) 