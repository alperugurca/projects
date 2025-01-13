#importing the necessary libraries
import nltk #natural language toolkit
from nltk.util import ngrams #ngrams is a function that returns a list of n-grams from a given text
from nltk.tokenize import word_tokenize #word_tokenize is a function that tokenizes a given text into words

from collections import Counter #counter is a dictionary subclass for counting hashable objects



#make a data set

corpus =  ["I love programming in python",
           "python is a great language",
           "I love the python programming language",
           "python is easy to learn",
           "I love the python programming language"
            ]

"""
Problem description:
Make a language model that can predict the next word in a sentence.
Using n-gram models, we can predict the next word in a sentence by looking at the previous n words.
one word is a unigram, two words is a bigram, three words is a trigram, etc.

example: I ...(love)
"""




#tokenize the data set
tokens = [word_tokenize(sentence.lower()) for sentence in corpus]


#bigram model

bigrams = []
for token_list in tokens:
    bigrams.extend(ngrams(token_list, 2))

bigrams_freq = Counter(bigrams)

print(bigrams_freq)
#trigram model

trigrams = []
for token_list in tokens:
    trigrams.extend(ngrams(token_list,3))

trigram_freq = Counter(trigrams)

print(trigram_freq)


#model testing

bigram = ('i', 'love') # target bigram

prob_the = trigram_freq[('i', 'love', 'the')]/bigrams_freq[('i', 'love')]

print(f'I love -the- probability is: {prob_the}')


prob_programming = trigram_freq[('i', 'love', 'programming')]/bigrams_freq[('i', 'love')]

print(f'I love -programming- probability is: {prob_programming}')