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

from tensorflow import keras
from keras import Input
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

def create_mlp(name, n_hidden_1, n_hidden_2, num_features, num_classes, gaussian_init=False, gaussian_mean=-1, kernel_reg=None, a_reg=-1, dropout_layers=False, dropout_prob=-1):
  
  kwargs = dict()
  
  if gaussian_init:
    kwargs["kernel_initializer"] = initializers.RandomNormal(mean=gaussian_mean)

  if kernel_reg == "l2":
    kwargs["kernel_regularizer"] = regularizers.l2(a_reg)

  if kernel_reg == "l1":
    kwargs["kernel_regulizer"] = regularizers.l1(a_reg)

  mlp = keras.Sequential(name=name)

  mlp.add(Input(shape=(num_features,)))
  mlp.add(keras.layers.Dense(name="hidden_layer_1", units=n_hidden_1, activation="relu", **kwargs))
  if dropout_layers == True: mlp.add(keras.layers.Dropout(dropout_prob))
  mlp.add(keras.layers.Dense(name="hidden_layer_2", units=n_hidden_2, activation="relu", **kwargs))
  if dropout_layers == True: mlp.add(keras.layers.Dropout(dropout_prob))
  mlp.add(keras.layers.Dense(name="output_layer", units=num_classes, activation="softmax", **kwargs))
  if dropout_layers == True: mlp.add(keras.layers.Dropout(dropout_prob))
  
  mlp.summary()

  return mlp

import matplotlib.pyplot as plt

def disp_results(mlp, X_train, y_train, X_test, y_test, history):

  # TODO add comments and titles
  print("Evaluation on training data:")
  train_results = mlp.evaluate(X_train, y_train)
  print("Evaluation on testing data:")
  test_results = mlp.evaluate(X_test, y_test)

  loss_train = history.history["loss"]
  loss_test = history.history["val_loss"]

  accuracy_train = history.history["accuracy"]
  accuracy_test = history.history["val_accuracy"]

  plt.plot(loss_train, label="loss train")
  plt.plot(loss_test, label="loss test")
  plt.legend()
  plt.show()

  plt.plot(accuracy_train, label="accuracy train")
  plt.plot(accuracy_test, label="accuracy test")
  plt.legend()
  plt.show()

