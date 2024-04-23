import keras
import numpy as np
from keras import Input
from keras import models
from keras import layers
from keras import optimizers
from keras.models import Model
from keras import applications
from keras import backend as k
import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoard,
    EarlyStopping,
)


train_datagen = ImageDataGenerator(rescale=1.0 / 255)


validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)


train_generator = train_datagen.flow_from_directory(
    "/content/drive/MyDrive/AMITY/Deep Learning (codes)/Data/plant_village/plant_village/train/",
    target_size=(128, 128),
    batch_size=16,
    class_mode="categorical",
)


validation_generator = validation_datagen.flow_from_directory(
    "/content/drive/MyDrive/AMITY/Deep Learning (codes)/Data/plant_village/plant_village/val/",
    target_size=(128, 128),
    batch_size=16,
    class_mode="categorical",
    shuffle=False,
)

test_generator = test_datagen.flow_from_directory(
    "/content/drive/MyDrive/AMITY/Deep Learning (codes)/Data/plant_village/plant_village/test/",
    target_size=(128, 128),
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
)


plt.figure(figsize=(16, 16))
for i in range(1, 17):
    plt.subplot(4, 4, i)
    img, label = test_generator.next()

    plt.imshow(img[0])
plt.show()


img, label = test_generator.next()
img[0].shape


from tensorflow.keras.applications.vgg16 import VGG16


base_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

base_model.summary()


flatten_layer = layers.GlobalAveragePooling2D()


prediction_layer = layers.Dense(4, activation="softmax")

model = models.Sequential([base_model, flatten_layer, prediction_layer])

model.summary()


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["acc"],
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1,
)


model.save("VGG16_plant_deseas.h5")
print("Saved model to disk")


model = models.load_model("VGG16_plant_deseas.h5")
print("Model is loaded")


model.save_weights("cnn_classification.h5")


model.load_weights("cnn_classification.h5")


train_acc = history.history["acc"]
val_acc = history.history["val_acc"]
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]


epochs = range(len(train_acc))
plt.plot(epochs, train_acc, "b", label="Training Accuracy")
plt.plot(epochs, val_acc, "g", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.grid()
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs, train_loss, "b", label="Training Loss")
plt.plot(epochs, val_loss, "g", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.grid()
plt.legend()
plt.show()


fnames = test_generator.filenames


ground_truth = test_generator.classes


label2index = test_generator.class_indices


idx2label = dict((v, k) for k, v in label2index.items())


predictions = model.predict_generator(
    test_generator, steps=test_generator.samples / test_generator.batch_size, verbose=1
)
predicted_classes = np.argmax(predictions, axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors), test_generator.samples))


accuracy = ((test_generator.samples - len(errors)) / test_generator.samples) * 100
accuracy


from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

cm = confusion_matrix(y_true=ground_truth, y_pred=predicted_classes)
cm = np.array(cm)

cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cmn,
    annot=True,
    fmt=".2f",
    xticklabels=label2index,
    yticklabels=label2index,
    cmap="YlGnBu",
)
plt.ylabel("Actual", fontsize=15)
plt.xlabel("Predicted", fontsize=15)
plt.show(block=False)


from sklearn.metrics import classification_report

print(classification_report(ground_truth, predicted_classes, target_names=label2index))


from keras import applications


base_model = applications.InceptionV3(
    weights="imagenet", include_top=False, input_shape=(128, 128, 3)
)
base_model.trainable = False

base_model.summary()


flatten_layer = layers.GlobalAveragePooling2D()


prediction_layer = layers.Dense(4, activation="softmax")

model = models.Sequential(
    [
        base_model,
        flatten_layer,
        prediction_layer,
    ]
)

model.summary()


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["acc"],
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1,
)


model.save("InceptionNet_plant_deseas.h5")
print("Saved model to disk")


train_acc = history.history["acc"]
val_acc = history.history["val_acc"]
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]


epochs = range(len(train_acc))
plt.plot(epochs, train_acc, "b", label="Training Accuracy")
plt.plot(epochs, val_acc, "g", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.grid()
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs, train_loss, "b", label="Training Loss")
plt.plot(epochs, val_loss, "g", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.grid()
plt.legend()
plt.show()


fnames = test_generator.filenames


ground_truth = test_generator.classes


label2index = test_generator.class_indices


idx2label = dict((v, k) for k, v in label2index.items())


predictions = model.predict_generator(
    test_generator, steps=test_generator.samples / test_generator.batch_size, verbose=1
)
predicted_classes = np.argmax(predictions, axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors), test_generator.samples))


accuracy = ((test_generator.samples - len(errors)) / test_generator.samples) * 100
accuracy


from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

cm = confusion_matrix(y_true=ground_truth, y_pred=predicted_classes)
cm = np.array(cm)

cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cmn,
    annot=True,
    fmt=".2f",
    xticklabels=label2index,
    yticklabels=label2index,
    cmap="YlGnBu",
)
plt.ylabel("Actual", fontsize=15)
plt.xlabel("Predicted", fontsize=15)
plt.show(block=False)


from sklearn.metrics import classification_report

print(classification_report(ground_truth, predicted_classes, target_names=label2index))


from keras import applications


base_model = applications.ResNet50(
    weights="imagenet", include_top=False, input_shape=(128, 128, 3)
)
base_model.trainable = False

base_model.summary()


flatten_layer = layers.GlobalAveragePooling2D()


prediction_layer = layers.Dense(4, activation="softmax")

model = models.Sequential(
    [
        base_model,
        flatten_layer,
        prediction_layer,
    ]
)

model.summary()


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["acc"],
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1,
)


model.save("ResNet_plant_deseas.h5")
print("Saved model to disk")


train_acc = history.history["acc"]
val_acc = history.history["val_acc"]
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]


epochs = range(len(train_acc))
plt.plot(epochs, train_acc, "b", label="Training Accuracy")
plt.plot(epochs, val_acc, "g", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.grid()
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs, train_loss, "b", label="Training Loss")
plt.plot(epochs, val_loss, "g", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.grid()
plt.legend()
plt.show()


fnames = test_generator.filenames


ground_truth = test_generator.classes


label2index = test_generator.class_indices


idx2label = dict((v, k) for k, v in label2index.items())


predictions = model.predict_generator(
    test_generator, steps=test_generator.samples / test_generator.batch_size, verbose=1
)
predicted_classes = np.argmax(predictions, axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors), test_generator.samples))


accuracy = ((test_generator.samples - len(errors)) / test_generator.samples) * 100
accuracy


from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

cm = confusion_matrix(y_true=ground_truth, y_pred=predicted_classes)
cm = np.array(cm)

cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cmn,
    annot=True,
    fmt=".2f",
    xticklabels=label2index,
    yticklabels=label2index,
    cmap="YlGnBu",
)
plt.ylabel("Actual", fontsize=15)
plt.xlabel("Predicted", fontsize=15)
plt.show(block=False)


from sklearn.metrics import classification_report

print(classification_report(ground_truth, predicted_classes, target_names=label2index))


vgg_model = VGG16(include_top=False, input_shape=(64, 64, 3))

for layer in vgg_model.layers:
    layer.trainable = False

flat1 = Flatten()(vgg_model.layers[-1].output)
class1 = Dense(256, activation="relu")(flat1)
output = Dense(4, activation="sigmoid")(class1)

model = Model(inputs=vgg_model.inputs, outputs=output)
