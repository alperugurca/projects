# Part of speec POS tagging

# Hidden Markov Model HMM

#import libraries
import nltk
from nltk.tag import hmm

#example training data 
train_data = [
    [("I","PRP"),("am","VBP"),("a","DT"),("teacher","NN")],
    [("You","PRP"),("are","VBP"),("a","DT"),("student","NN")],
]

#train the model

trainer= hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)


# new sentence and POS tagging

test_sentence = "I am a student".split()

hmm_tagger.tag(test_sentence)

print(f'The sentence is: {test_sentence}')


test_sentence2 = "He is a driver".split()

tags2 = hmm_tagger.tag(test_sentence2)

print(f'The sentence is: {test_sentence2}')
print(f'The tags are: {tags2}')
