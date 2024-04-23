import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import (
    pad_sequences,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import re
from keras.layers import SimpleRNN


data = pd.read_csv(
    "/content/drive/MyDrive/AMITY/Deep Learning (codes)/Data/IMDB Dataset.csv"
)
print(data)


import nltk

nltk.download("stopwords")
english_stops = set(stopwords.words("english"))


def load_dataset():
    df = pd.read_csv(
        "/content/drive/MyDrive/AMITY/Deep Learning (codes)/Data/IMDB Dataset.csv"
    )
    x_data = df["review"]
    y_data = df["sentiment"]

    x_data = x_data.replace({"<.*?>": ""}, regex=True)
    x_data = x_data.replace({"[^A-Za-z]": " "}, regex=True)
    x_data = x_data.apply(
        lambda review: [w for w in review.split() if w not in english_stops]
    )
    x_data = x_data.apply(lambda review: [w.lower() for w in review])

    y_data = y_data.replace("positive", 1)
    y_data = y_data.replace("negative", 0)

    return x_data, y_data


x_data, y_data = load_dataset()

print("Reviews")
print(x_data, "\n")
print("Sentiment")
print(y_data)


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

print("Train Set")
print(x_train, "\n")
print(x_test, "\n")
print("Test Set")
print(y_train, "\n")
print(y_test)


def get_max_length():
    review_length = []
    for review in x_train:
        review_length.append(len(review))

    return int(np.ceil(np.mean(review_length)))


token = Tokenizer(lower=False)
token.fit_on_texts(x_train)
x_train = token.texts_to_sequences(x_train)
x_test = token.texts_to_sequences(x_test)

max_length = get_max_length()

x_train = pad_sequences(x_train, maxlen=max_length, padding="post", truncating="post")
x_test = pad_sequences(x_test, maxlen=max_length, padding="post", truncating="post")

total_words = len(token.word_index) + 1
print("Total Words:", total_words)

print("Encoded X Train\n", x_train, "\n")
print("Encoded X Test\n", x_test, "\n")
print("Maximum review length: ", max_length)


rnn = Sequential()

rnn.add(Embedding(total_words, 32, input_length=max_length))
rnn.add(
    SimpleRNN(
        64,
        input_shape=(total_words, max_length),
        return_sequences=False,
        activation="relu",
    )
)
rnn.add(Dense(1, activation="sigmoid"))

print(rnn.summary())
rnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


history = rnn.fit(x_train, y_train, epochs=20, batch_size=128, verbose=1)


model = rnn.save("rnn.h5")
loaded_model = load_model("rnn.h5")


y_pred = rnn.predict(x_test, batch_size=128)
print(y_pred)
print(y_test)
for i in range(len(y_pred)):
    if y_pred[i] > 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0

true = 0
for i, y in enumerate(y_test):
    if y == y_pred[i]:
        true += 1

print("Correct Prediction: {}".format(true))
print("Wrong Prediction: {}".format(len(y_pred) - true))
print("Accuracy: {}".format(true / len(y_pred) * 100))


review = str(input("Movie Review: "))


regex = re.compile(r"[^a-zA-Z\s]")
review = regex.sub("", review)
print("Cleaned: ", review)

words = review.split(" ")
filtered = [w for w in words if w not in english_stops]
filtered = " ".join(filtered)
filtered = [filtered.lower()]

print("Filtered: ", filtered)


tokenize_words = token.texts_to_sequences(filtered)
tokenize_words = pad_sequences(
    tokenize_words, maxlen=max_length, padding="post", truncating="post"
)
print(tokenize_words)


result = rnn.predict(tokenize_words)
print(result)


if result >= 0.7:
    print("positive")
else:
    print("negative")
