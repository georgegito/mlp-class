import tensorflow as tf
# from tensorflow.keras import Model, layers
from tensorflow.keras.datasets import mnist
# from tensorflow.keras import backend as K
from tensorflow.keras import initializers

import numpy as np
from torch.nn.modules import activation

# MNIST dataset params
num_classes = 10 # 0-9 digits
num_features = 784 # img shape: 28*28

# network params
n_hidden_1 = 128
n_hidden_2 = 256

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# convert to float32
x_train = np.array(x_train, np.float32)
# vectorize images
x_train = x_train.reshape([-1, num_features])
# normalize images values from [0, 255] to [0, 1]
x_train = x_train / 255.

# create model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(784, )),
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializers.RandomNormal(mean=10)),
    tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializers.RandomNormal(mean=10)),
    tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializers.RandomNormal(mean=10))
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])

# model.summary()

# train model
history = model.fit(x_train, y_train, batch_size=256, validation_split=0.2, epochs=100, verbose=1)

# plots
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.title('Performance on training and validation sets')
plt.show()

plt.figure(2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.title('Learning curves on training and validation sets')
plt.show()