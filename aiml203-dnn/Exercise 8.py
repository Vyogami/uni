import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("/content/drive/MyDrive/AMITY/Deep Learning (codes)/Data/data.csv")
df.head()


classes = ["Hate Speech", "Offensive Language", "None"]


df.drop(
    ["count", "hate_speech", "offensive_language", "neither", "Unnamed: 0"],
    axis=1,
    inplace=True,
)


df.head()


df.shape


labels = df["class"]
unique, counts = np.unique(labels, return_counts=True)
values = list(zip(unique, counts))
plt.bar(classes, counts)
for i in values:
    print(classes[i[0]], " : ", i[1])
plt.show()


hate_tweets = df[df["class"] == 0]
offensive_tweets = df[df["class"] == 1]
neither = df[df["class"] == 2]
print(hate_tweets.shape)
print(offensive_tweets.shape)
print(neither.shape)


for i in range(3):
    hate_tweets = pd.concat([hate_tweets, hate_tweets], ignore_index=True)
neither = pd.concat([neither, neither, neither], ignore_index=True)
offensive_tweets = offensive_tweets.iloc[0:12000, :]
print(hate_tweets.shape)
print(offensive_tweets.shape)
print(neither.shape)


df = pd.concat([hate_tweets, offensive_tweets, neither], ignore_index=True)
df.shape


labels = df["class"]
unique, counts = np.unique(labels, return_counts=True)
values = list(zip(unique, counts))
plt.bar(classes, counts)
for i in values:
    print(classes[i[0]], " : ", i[1])
plt.show()


df.head()


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

nltk.download("wordnet")
nltk.download("stopwords")


d = {
    "luv": "love",
    "wud": "would",
    "lyk": "like",
    "wateva": "whatever",
    "ttyl": "talk to you later",
    "kul": "cool",
    "fyn": "fine",
    "omg": "oh my god!",
    "fam": "family",
    "bruh": "brother",
    "cud": "could",
    "fud": "food",
    "u": "you",
    "ur": "your",
    "bday": "birthday",
    "bihday": "birthday",
}


stop_words = set(stopwords.words("english"))
stop_words.add("rt")
stop_words.remove("not")
lemmatizer = WordNetLemmatizer()
giant_url_regex = (
    "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|" "[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
mention_regex = "@[\w\-]+"


def clean_text(text):
    text = re.sub('"', "", text)
    text = re.sub(mention_regex, " ", text)
    text = re.sub(giant_url_regex, " ", text)
    text = text.lower()
    text = re.sub("hm+", "", text)
    text = re.sub("[^a-z]+", " ", text)
    text = text.split()
    text = [word for word in text if not word in stop_words]
    text = [d[word] if word in d else word for word in text]
    text = [lemmatizer.lemmatize(token) for token in text]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = " ".join(text)
    return text


df["processed_tweets"] = df.tweet.apply(lambda x: clean_text(x))
df.head()


x = df.processed_tweets
y = df["class"]
print(x.shape)
print(y.shape)


word_unique = []
for i in x:
    for j in i.split():
        word_unique.append(j)
unique, counts = np.unique(word_unique, return_counts=True)
print("The total words in the tweets are : ", len(word_unique))
print("The total UNIQUE words in the tweets are : ", len(unique))


tweets_length = []
for i in x:
    tweets_length.append(len(i.split()))
print("The Average Length tweets are : ", np.mean(tweets_length))
print("The max length of tweets is : ", np.max(tweets_length))
print("The min length of tweets is : ", np.min(tweets_length))


tweets_length = pd.DataFrame(tweets_length)


col = list(zip(unique, counts))
col = sorted(col, key=lambda x: x[1], reverse=True)
col = pd.DataFrame(col)
print("Top 20 Occuring Words with their frequency are:")
col.iloc[:20, :]


from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer(max_features=8000)


vectorizer.fit(x)


print(len(vectorizer.vocabulary_))
print(vectorizer.idf_.shape)


x_tfidf = vectorizer.transform(x).toarray()
print(x_tfidf.shape)


from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


num_words = 8000
embed_dim = 32
tokenizer = Tokenizer(num_words=num_words, oov_token="<oov>")
tokenizer.fit_on_texts(x)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(x)
length = []
for i in sequences:
    length.append(len(i))
print(len(length))
print("Mean is: ", np.mean(length))
print("Max is: ", np.max(length))
print("Min is: ", np.min(length))


pad_length = 24
sequences = pad_sequences(
    sequences, maxlen=pad_length, truncating="pre", padding="post"
)
sequences.shape


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sequences, y, test_size=0.05)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


from keras.layers import Dense, Embedding, Dropout, Activation, Flatten, SimpleRNN
from keras.layers import GlobalMaxPool1D
from keras.models import Model, Sequential
import tensorflow as tf


recall = tf.keras.metrics.Recall()
precision = tf.keras.metrics.Precision()

model = Sequential(
    [
        Embedding(num_words, embed_dim, input_length=pad_length),
        SimpleRNN(8, return_sequences=True),
        GlobalMaxPool1D(),
        Dense(20, activation="relu", kernel_initializer="he_uniform"),
        Dropout(0.25),
        Dense(3, activation="softmax"),
    ]
)
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

model.summary()


history = model.fit(x=x_train, y=y_train, epochs=5, validation_split=0.05)


evaluate = model.evaluate(x_test, y_test)


print("Test Acuracy is : {:.2f} %".format(evaluate[1] * 100))
print("Test Loss is : {:.4f}".format(evaluate[0]))


predictions = model.predict(x_test)


predict = []
for i in predictions:
    predict.append(np.argmax(i))


from sklearn import metrics

cm = metrics.confusion_matrix(predict, y_test)
acc = metrics.accuracy_score(predict, y_test)


print("The Confusion matrix is: \n", cm)


print(acc * 100)


from sklearn import metrics

print(metrics.classification_report(y_test, predict))


from tensorflow.keras.layers import Embedding, LSTM, Dense


EMBED_DIM = 32
LSTM_OUT = 64

model = Sequential()
model.add(Embedding(num_words, EMBED_DIM, input_length=pad_length))
model.add(LSTM(LSTM_OUT))
model.add(Dense(3, activation="softmax"))
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print(model.summary())


history = model.fit(x=x_train, y=y_train, epochs=10, validation_split=0.05)


evaluate = model.evaluate(x_test, y_test)


print("Test Acuracy is : {:.2f} %".format(evaluate[1] * 100))
print("Test Loss is : {:.4f}".format(evaluate[0]))


predictions = model.predict(x_test)


predict = []
for i in predictions:
    predict.append(np.argmax(i))


from sklearn import metrics

cm = metrics.confusion_matrix(predict, y_test)
acc = metrics.accuracy_score(predict, y_test)


print("The Confusion matrix is: \n", cm)


print(acc * 100)


from sklearn import metrics

print(metrics.classification_report(y_test, predict))
