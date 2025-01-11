import nltk 
nltk.download("wordnet") # wordnet: data for lemmatization
from nltk.stem import PorterStemmer # stemming function

    
# create porter stemmer object
stemmer = PorterStemmer()

words = ["running", "runner", "ran", "runs", "better", "go", "went"]

# stemming with porter stemmer
stems = [stemmer.stem(w) for w in words]
print(f"Stems: {stems}")

# %% lemmatization

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = ["running", "runner", "ran", "runs", "better", "go", "went"]
lemmas = [lemmatizer.lemmatize(w, pos="v") for w in words]
print(f"Lemmas: {lemmas}")



















