# ---------------------------------------------------------------------------- #
#                                  metrics.py                                  #
# ---------------------------------------------------------------------------- #

from keras import backend as K

# ---------------------------------- recall ---------------------------------- #

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    _recall = true_positives / (possible_positives + K.epsilon())
    return _recall

# --------------------------------- precision -------------------------------- #

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    _precision = true_positives / (predicted_positives + K.epsilon())
    return _precision

# ------------------------------------ f1 ------------------------------------ #

def f1(y_true, y_pred):
  _precision = precision(y_true, y_pred)
  _recall = precision(y_true, y_pred)
  return 2*((_precision * _recall) / (_precision + _recall + K.epsilon()))

# ---------------------------------------------------------------------------- #