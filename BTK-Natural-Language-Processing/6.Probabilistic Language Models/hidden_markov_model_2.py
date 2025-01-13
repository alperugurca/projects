#import libraries
import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000



# read data
nltk.download('conll2000')
train_data = conll2000.tagged_sents("train.txt")
test_data = conll2000.tagged_sents("test.txt")

print(f'Train data: {train_data[:1]}')

# train hmm model

trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)


# new sentence and test

test_sentence = "I like going to the school".split()

tags = hmm_tagger.tag(test_sentence)
print(f'The sentence is: {test_sentence}')
print(f'The tags are: {tags}')