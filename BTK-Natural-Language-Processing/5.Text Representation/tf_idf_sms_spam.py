# import library
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# read dataset
df = pd.read_csv("sms_spam.csv")

# clean data

# tfidf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df.text)

# inspect feature names
feature_names = vectorizer.get_feature_names_out()
tfidf_score = X.mean(axis=0).A1 # each word's average tf-idf score

# create a df with tfidf scores
df_tfidf = pd.DataFrame({"word":feature_names, "tfidf_score": tfidf_score})

# sort scores and inspect results
df_tfidf_sorted = df_tfidf.sort_values(by="tfidf_score", ascending=False)
print(df_tfidf_sorted.head(10))