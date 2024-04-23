import numpy as np
import pandas as pd


bird_data = pd.read_csv("bird.csv", delimiter=",")
bird_data.head(5)


bird_data.columns


bird_data = bird_data.set_index("id")
bird_data.head()


bird_data.shape


bird_data.info()


bird_data.isna().sum()


bird_data.dropna(how="any", inplace=True)


bird_data.isna().sum()


bird_data.shape


bird_data.nunique()


bird_data["type"].unique()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
bird_data[["type"]] = bird_data[["type"]].apply(le.fit_transform)


bird_data.head()


y = bird_data["type"]
X = bird_data.drop(["type"], axis=1)


X.columns


y


y.shape


from keras.utils import np_utils

num_classes = 6
y = np_utils.to_categorical(y, num_classes)
y


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


print("Shape of the X_train", X_train.shape)
print("Shape of the X_test", X_test.shape)
print("Shape of the y_train", y_train.shape)
print("Shape of the y_test", y_test.shape)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()

classifier.add(
    Dense(units=8, kernel_initializer="uniform", activation="relu", input_dim=10)
)


classifier.add(Dense(units=16, kernel_initializer="uniform", activation="relu"))


classifier.add(Dense(units=32, kernel_initializer="uniform", activation="relu"))


classifier.add(Dense(units=6, kernel_initializer="uniform", activation="softmax"))


classifier.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)


classifier.fit(X_train, y_train, batch_size=16, epochs=800, verbose=1)


score, acc = classifier.evaluate(X_train, y_train, batch_size=10)
print("Train score:", score)
print("Train accuracy:", acc)

print("*" * 20)
score, acc = classifier.evaluate(X_test, y_test, batch_size=10)
print("Test score:", score)
print("Test accuracy:", acc)


pred = classifier.predict(X_test)
print("Y_pred:", pred)
print("*****************")
y_pred = np.argmax(pred, axis=1)
print("Y_pred:", y_pred)
print("*****************")
print("Y_test:", y_test)
y_true = np.argmax(y_test, axis=1)
print("*****************")
print("Y_test:", y_true)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
target_names = ["P", "R", "SO", "SW", "T", "W"]


import matplotlib.pyplot as plt
import seaborn as sns


p = sns.heatmap(
    pd.DataFrame(cm),
    annot=True,
    xticklabels=target_names,
    yticklabels=target_names,
    cmap="YlGnBu",
    fmt="g",
)
plt.title("Confusion matrix", y=1.1)
plt.ylabel("Actual label")
plt.xlabel("Predicted label")


from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=target_names))


from sklearn.metrics import roc_curve, auc
from itertools import cycle

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


for i in range(6):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label="ROC curve (area = %0.2f)" % roc_auc[i])
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()


fpr = dict()
tpr = dict()
roc_auc = dict()
lw = 2
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(["blue", "green", "red", "darkorange", "olive", "purple"])
for i, color in zip(range(6), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=2,
        label="AUC = {1:0.4f}" "".format(i, roc_auc[i]),
    )
plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=15)
plt.ylabel("True Positive Rate", fontsize=15)

plt.legend(loc="lower right")
plt.show()


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(
        Dense(units=8, kernel_initializer="uniform", activation="relu", input_dim=10)
    )
    classifier.add(Dense(units=16, kernel_initializer="uniform", activation="relu"))
    classifier.add(Dense(units=32, kernel_initializer="uniform", activation="relu"))
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="sigmoid"))
    classifier.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return classifier


classifier = KerasClassifier(build_fn=build_classifier)
parameters = {
    "batch_size": [16, 32],
    "epochs": [800, 1000],
    "optimizer": ["adam", "rmsprop"],
}
grid_search = GridSearchCV(
    estimator=classifier, param_grid=parameters, scoring="accuracy", cv=10
)
grid_search = grid_search.fit(X_train, y_train, verbose=1)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
