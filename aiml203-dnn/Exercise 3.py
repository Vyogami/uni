import numpy as np
import pandas as pd


Powerplant_data = pd.read_excel("Folds5x2_pp.xlsx")
Powerplant_data.head(5)


Powerplant_data.columns


Powerplant_data.shape


Powerplant_data.info()


Powerplant_data.isna().sum()


Powerplant_data.nunique()


X = Powerplant_data.iloc[:, :-1].values
y = Powerplant_data.iloc[:, -1].values


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=True
)


print("Shape of the X_train", X_train.shape)
print("Shape of the X_test", X_test.shape)
print("Shape of the X_val", X_val.shape)
print("Shape of the y_train", y_train.shape)
print("Shape of the y_test", y_test.shape)
print("Shape of the y_val", y_val.shape)


from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()

classifier.add(
    Dense(units=8, kernel_initializer="uniform", activation="relu", input_dim=4)
)


classifier.add(Dense(units=16, kernel_initializer="uniform", activation="relu"))


classifier.add(Dense(units=32, kernel_initializer="uniform", activation="relu"))


classifier.add(Dense(units=1, kernel_initializer="uniform"))


classifier.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["MeanSquaredLogarithmicError"]
)


model = classifier.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=200,
    validation_data=(X_val, y_val),
    shuffle=True,
)


y_pred = classifier.predict(X_test)
np.set_printoptions(precision=2)
print(
    np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
)


import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")


import sklearn.metrics
from math import sqrt

mae_no = sklearn.metrics.mean_absolute_error(y_test, classifier.predict(X_test))
mse_no = sklearn.metrics.mean_squared_error(y_test, classifier.predict(X_test))
rms = sqrt(sklearn.metrics.mean_squared_error(y_test, classifier.predict(X_test)))


print("Mean Absolute Error     :", mae_no)
print("Mean Square Error       :", mse_no)
print("Root Mean Square Error:", rms)
