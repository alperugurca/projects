# Classification problem sentiment analysis
# import libraries
from nltk.classify import MaxentClassifier



# load data
train_data = [
    ({"love": True, "amazing": True, "happy": True, "terrible": False}, "positive"),
    ({"hate": True, "terrible": True}, "negative"),
    ({"joy": True, "happy": True, "hate": False}, "positive"),
    ({"sad": True, "depresssed": True, "love": False}, "negative"),
]


# train maximum entropy model

classifier = MaxentClassifier.train(train_data, max_iter=10)


# test maximum entropy model


test_sentence = "I do hate terrible this movie"

features = {word: (word in test_sentence.lower().split()) for word in ["love", "amazing", "happy", "terrible", "joy", "sad", "depresssed", "hate", "terrible"]}

label = classifier.classify(features)
print(f"Result: {label}")