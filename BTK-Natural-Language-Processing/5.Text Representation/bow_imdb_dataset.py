# import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter

# import dataset
df = pd.read_csv("IMDB Dataset.csv")

# text and labels
documents = df["review"]
labels = df["sentiment"] # positive veya negative

# text cleaning
def clean_text(text):
    
    # lowercase
    text = text.lower()
    
    # remove numbers
    text = re.sub(r"\d+", "", text)
    
    # remove special characters
    text = re.sub(r"[^\w\s]", "", text)
    
    # remove short words
    text = " ".join([word for word in text.split() if len(word) > 2])
    
    return text # return cleaned text

# clean text
cleaned_doc = [clean_text(row) for row in documents]


# %% bow
# define vectorizer
vectorizer = CountVectorizer()

# convert text to numerical vectors
X = vectorizer.fit_transform(cleaned_doc[:75])

# show feature names
feature_names = vectorizer.get_feature_names_out()

# show vector representation
vektor_temsili2 = X.toarray()
print(f"Vektor temsili: {vektor_temsili2}")

df_bow = pd.DataFrame(vektor_temsili2, columns = feature_names)

# show word frequencies
word_counts = X.sum(axis=0).A1
word_freq = dict(zip(feature_names, word_counts))

# print first 5 words
most_common_5_words = Counter(word_freq).most_common(5)
print(f"most_common_5_words: {most_common_5_words}")


print(documents)



















# %%
