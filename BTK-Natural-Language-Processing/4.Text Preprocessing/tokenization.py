import nltk # Natural Language Toolkit

nltk.download("punkt")
nltk.download("punkt_tab")
import time
time.sleep(2)
text = "Hello, World! How are you? Hello, hi ..."



word_tokens = nltk.word_tokenize(text) # Tokenize the text into words
print(word_tokens)



sentence_tokens = nltk.sent_tokenize(text) # Tokenize the text into sentences
print(sentence_tokens)

