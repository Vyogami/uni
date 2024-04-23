import numpy as np
import pandas as pd


churn_data = pd.read_csv("Churn_Modelling.csv", delimiter=",")
churn_data.head(5)


churn_data.columns


churn_data = churn_data.set_index("RowNumber")
churn_data.head()


churn_data.shape


churn_data.info()


churn_data.isna().sum()


churn_data.nunique()


churn_data.drop(["CustomerId", "Surname"], axis=1, inplace=True)


churn_data.head()


churn_data.shape


from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

df = churn_data.copy()


def plot_univariate(col):
    if df[col].nunique() > 2:
        plt.figure(figsize=(10, 7))
        h = 0.15
        rot = 90
    else:
        plt.figure(figsize=(6, 6))
        h = 0.5
        rot = 0
    plot = sns.countplot(x=df[col], palette="pastel")

    for bars in plot.containers:
        for p in bars:
            plot.annotate(
                format(p.get_height()),
                (p.get_x() + p.get_width() * 0.5, p.get_height()),
                ha="center",
                va="bottom",
            )
            plot.annotate(
                f"{p.get_height()*100/df[col].shape[0] : .1f}%",
                (p.get_x() + p.get_width() * 0.5, h * p.get_height()),
                ha="center",
                va="bottom",
                rotation=rot,
            )


def plot_bivariate(col, hue):
    if df[col].nunique() > 5:
        plt.figure(figsize=(20, 10))
        rot = 90
    else:
        plt.figure(figsize=(10, 7))
        rot = 0

    def percentage(ax):
        heights = [[p.get_height() for p in bars] for bars in ax.containers]
        for bars in ax.containers:
            for i, p in enumerate(bars):
                total = sum(group[i] for group in heights)
                percentage = 100 * p.get_height() / total
                ax.annotate(
                    format(p.get_height()),
                    (p.get_x() + p.get_width() * 0.5, 0.8 * p.get_height()),
                    ha="center",
                    va="bottom",
                    rotation=0,
                )
                if percentage > 25.0:
                    percentage = f"{percentage:.1f}%"
                    ax.annotate(
                        percentage,
                        (p.get_x() + p.get_width() * 0.5, 0.25 * p.get_height()),
                        ha="center",
                        va="center",
                        rotation=rot,
                    )

    plot = sns.countplot(x=df[col], hue=df[hue], palette="pastel")
    percentage(plot)


def spearman(df, hue):
    feature = []
    correlation = []
    result = []
    for col in df.columns:
        corr, p = stats.spearmanr(df[col], df[hue])
        feature.append(col)
        correlation.append(corr)
        alpha = 0.05
        if p > alpha:
            result.append("No correlation (fail to reject H0)")
        else:
            result.append("Some correlation (reject H0)")
    c = pd.DataFrame(
        {
            "Feature Name": feature,
            "correlation coefficient": correlation,
            "Inference": result,
        }
    )
    display(c)


plot_univariate("Age")


plot_bivariate("Age", "Exited")


spearman(churn_data, "Age")


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
churn_data[["Geography", "Gender"]] = churn_data[["Geography", "Gender"]].apply(
    le.fit_transform
)


churn_data.head()


y = churn_data.Exited
X = churn_data.drop(["Exited"], axis=1)


X.columns


y


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)


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


classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))


classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


classifier.fit(X_train, y_train, batch_size=10, epochs=100, verbose=1)


score, acc = classifier.evaluate(X_train, y_train, batch_size=10)
print("Train score:", score)
print("Train accuracy:", acc)


y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5

print("*" * 20)
score, acc = classifier.evaluate(X_test, y_test, batch_size=10)
print("Test score:", score)
print("Test accuracy:", acc)


from sklearn.metrics import confusion_matrix

target_names = ["Retained", "Closed"]
cm = confusion_matrix(y_test, y_pred)
print(cm)


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

print(classification_report(y_test, y_pred, target_names=target_names))


from sklearn.metrics import roc_curve, auc

y_pred_proba = classifier.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr, tpr, label="AUC (area = %0.2f)" % roc_auc)
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.grid()
plt.legend(loc="lower right")
plt.title("ROC curve")
plt.show()


from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_pred_proba)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(
        Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=10)
    )
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
    classifier.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return classifier


classifier = KerasClassifier(build_fn=build_classifier)
parameters = {
    "batch_size": [16, 32],
    "epochs": [50, 100],
    "optimizer": ["adam", "rmsprop"],
}
grid_search = GridSearchCV(
    estimator=classifier, param_grid=parameters, scoring="accuracy", cv=2
)
grid_search = grid_search.fit(X_train, y_train, verbose=1)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


best_parameters


best_accuracy
