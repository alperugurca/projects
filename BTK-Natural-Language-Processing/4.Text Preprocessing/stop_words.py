import nltk

from nltk.corpus import stopwords 

nltk.download("stopwords") # stop words data set

# english stop words analysis (nltk)
stop_words_eng = set(stopwords.words("english"))

# example english text
text = "There are some examples of handling stop words from some texts."
text_list = text.split()
# if word is not in english stop words list (stop_words_eng), 
# add this word to filtered list
filtered_words = [word for word in text_list if word.lower() not in stop_words_eng]
print(f"filtered_words: {filtered_words}")

# %% turkce stop words analysis (nltk)
stop_words_tr = set(stopwords.words("turkish"))

# example turkish text
metin = "merhaba arkadaslar çok güzel bir ders işliyoruz. Bu ders faydalı mı?"
metin_list = metin.split()

filtered_words_tr = [word for word in metin_list if word.lower() not in stop_words_tr]
print(f"filtered_words_tr: {filtered_words_tr}")
# %% without library stop words removal

# create stop word list
tr_stopwords = ["için", "bu", "ile", "mu", "mi", "özel"]

# example turkish text
metin = "Bu bir denemedir. Amacımiz bu metinde bulunan özel karakterleri elemek mi acaba?"

filtered_words = [word for word in metin.split() if word.lower() not in tr_stopwords]
filtered_stop_words = set([word.lower() for word in metin.split() if word.lower() in tr_stopwords])

print(f"filtered_words: {filtered_words}")
print(f"filtered_stop_words: {filtered_stop_words}")















