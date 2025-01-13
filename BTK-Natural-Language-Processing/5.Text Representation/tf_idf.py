
# import libraries
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

# build documents
documents = [
    "Köpek çok tatlı bir hayvandır",
    "Köpek ve kuşlar çok tatlı hayvanlardır.",
    "Inekler süt üretirler."
    ]

# define vectorizer
tfidf_vectorizer = TfidfVectorizer()

# convert text to numerical vectors
X = tfidf_vectorizer.fit_transform(documents)

# inspect feature names
feature_names = tfidf_vectorizer.get_feature_names_out()

# inspect vector representation
vektor_temsili = X.toarray()
print(f"tf-idf: {vektor_temsili}")

df_tfidf = pd.DataFrame(vektor_temsili, columns=feature_names)

# inspect average tf-idf values
tf_idf = df_tfidf.mean(axis=0)


















