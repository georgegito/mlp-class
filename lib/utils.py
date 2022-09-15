# ---------------------------------------------------------------------------- #
#                                   utils.py                                   #
# ---------------------------------------------------------------------------- #

# --------------------------- display data summary --------------------------- #

import numpy as np

def data_summary(arr):

  shape = np.shape(arr)
  min = np.amin(arr)
  max = np.amax(arr)
  range = np.ptp(arr)
  variance = np.var(arr)
  sd = np.std(arr)
  print("Shape =", shape)
  print("Minimum =", min)
  print("Maximum =", max)
  print("Range =", range)
  print("Variance =", variance)
  print("Standard Deviation =", sd)
  print()

# ----------------------------- pre-process data ----------------------------- #

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def preprocess(X_train, y_train, X_test, y_test, num_classes, num_features, print_summary=True):

  # convert to float32
  X_train = np.array(X_train, np.float32)
  X_test = np.array(X_test, np.float32)

  # concatenate all data
  X = np.concatenate([X_train, X_test])
  y = np.concatenate([y_train, y_test])

  # shuffle data
  X, y = shuffle(X, y)

  # split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

  # vectorize images
  X_train = X_train.reshape([-1, num_features])
  X_test = X_test.reshape([-1, num_features])

  # normalize images values from [0, 255] to [0, 1]
  X_train = X_train / 255.
  X_test = X_test / 255.

  if print_summary:
    print("Training Inputs:")
    data_summary(X_train)

    print("Testing Inputs:")
    data_summary(X_test)

    print("Training Outputs:")
    data_summary(y_train)

    print("Testing Outputs:")
    data_summary(y_test)

  return X_train, y_train, X_test, y_test


# ------------------------- display training results ------------------------- #

import matplotlib.pyplot as plt

def disp_results(mlp, X_train, y_train, X_test, y_test, history):

  print("Evaluation on training data:")
  train_results = mlp.evaluate(X_train, y_train)
  print("Evaluation on testing data:")
  test_results = mlp.evaluate(X_test, y_test)

  loss_train = history.history["loss"]
  loss_test = history.history["val_loss"]

  accuracy_train = history.history["accuracy"]
  accuracy_test = history.history["val_accuracy"]

  plt.ion()
  plt.figure(figsize=(6, 2), dpi=600)
  plt.plot(loss_train, label="Training", linewidth=0.9)
  plt.plot(loss_test, label="Testing", linewidth=0.9)
  plt.title(mlp.name + " - Loss curves")
  plt.legend()
  plt.grid()
  plt.savefig("fig/" + mlp.name + "_Loss.jpg", dpi=1200)
  plt.show()

  plt.figure(figsize=(6, 2), dpi=600)
  plt.plot(accuracy_train, label="Training", linewidth=0.9)
  plt.plot(accuracy_test, label="Testing", linewidth=0.9)
  plt.title(mlp.name + " - Accuracy curves")
  plt.legend()
  plt.grid()
  plt.savefig("fig/" + mlp.name + "_Accuracy.jpg", dpi=1200)
  plt.show()

# ----------------------------- create mlp model ----------------------------- #

from tensorflow import keras
from keras import Input
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

def create_mlp(name, n_hidden_1, n_hidden_2, num_features, num_classes, gaussian_init=False, gaussian_mean=-1, kernel_reg=None, a_reg=-1, dropout_layers=False, dropout_prob=-1, print_summary=False):
  
  kwargs = dict()
  
  if gaussian_init:
    kwargs["kernel_initializer"] = initializers.RandomNormal(mean=gaussian_mean)

  if kernel_reg == "l2":
    kwargs["kernel_regularizer"] = regularizers.l2(a_reg)

  if kernel_reg == "l1":
    kwargs["kernel_regularizer"] = regularizers.l1(a_reg)

  mlp = keras.Sequential(name=name)

  mlp.add(Input(shape=(num_features,)))
  mlp.add(keras.layers.Dense(name="hidden_layer_1", units=n_hidden_1, activation="relu", **kwargs))
  if dropout_layers == True: mlp.add(keras.layers.Dropout(dropout_prob))
  mlp.add(keras.layers.Dense(name="hidden_layer_2", units=n_hidden_2, activation="relu", **kwargs))
  if dropout_layers == True: mlp.add(keras.layers.Dropout(dropout_prob))
  mlp.add(keras.layers.Dense(name="output_layer", units=num_classes, activation="softmax", **kwargs))
  if dropout_layers == True: mlp.add(keras.layers.Dropout(dropout_prob))
  
  if print_summary:
    mlp.summary()

  return mlp

# ------------------------------ predict classes ----------------------------- #

def predict_classes(model, X):

    return np.argmax(model.predict(X), axis=1)

# ---------------------------------------------------------------------------- #