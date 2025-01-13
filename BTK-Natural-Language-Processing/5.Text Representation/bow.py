# import count vectorizer
from sklearn.feature_extraction.text import CountVectorizer

# build dataset
documents = [
    "kedi bahçede",
    "kedi evde"]

# build vectorizer
vectorizer = CountVectorizer()

# convert text to numerical vectors
X = vectorizer.fit_transform(documents)


# build feature names [bahçede, evde, kedi]
feature_names = vectorizer.get_feature_names_out()
print(f"kelime kumesi: {feature_names}")

# build vector representation
vector_temsili = X.toarray()

print(f"vector_temsili: {vector_temsili}")


"""
kelime kumesi: ['bahçede' 'evde' 'kedi']
vector_temsili: 
    [[1 0 1]
     [0 1 1]]
"""



















