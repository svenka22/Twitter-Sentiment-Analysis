import en
from DataPreprocessing import DataPreprocessing
from nltk.collocations import BigramCollocationFinder
from nltk.metrics.association import BigramAssocMeasures
dataPreprocessing = DataPreprocessing()
from textblob import Word

#start extract_features
def get_bigrams1(tweet, score_fn=BigramAssocMeasures.chi_sq, n=200):
        bigramslist = []
        bigram_finder = BigramCollocationFinder.from_words(tweet)
        bigrams = bigram_finder.nbest(score_fn, n)
        for bigram in bigrams:
            bigramslist.append(' '.join(str(i) for i in bigram))
        print bigramslist
#end

# Checks the spelling of the word
# Returns a list of possible words from wordnet
def spellChecker(word):
    w = Word(word)
    final = list()
    corrected = w.spellcheck()
    for correct in corrected:
        prob = correct[1]
        if prob > 0.80:
            final.append(correct[0])
            return correct[0]
    return None

# 
# import re
# from itertools import groupby
#  
# def viterbi_segment(text):
#     probs, lasts = [1.0], [0]
#     for i in range(1, len(text) + 1):
#         prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
#                         for j in range(max(0, i - max_word_length), i))
#         probs.append(prob_k)
#         lasts.append(k)
#     words = []
#     i = len(text)
#     while 0 < i:
#         words.append(text[lasts[i]:i])
#         i = lasts[i]
#     words.reverse()
#     return words, probs[-1]
#  
# def word_prob(word): return dictionary.get(word, 0) / total
#  
# def words(text): return re.findall('[a-z]+', text.lower()) 
#  
# dictionary = dict((w, len(list(ws)))
#                     for w, ws in groupby(sorted(words(open('../data/big.txt').read()))))
# max_word_length = max(map(len, dictionary))
# total = float(sum(dictionary.values()))


if __name__ == '__main__':  print spellChecker('computre')