from __future__ import print_function

import keras
import sklearn
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Multiply
from keras.datasets import fashion_mnist

# Aditional Libs
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.dual import det, inv
from sklearn.model_selection import train_test_split

# the data, shuffled and split between train and test sets
import sklearn.metrics as metrics
import seaborn as sn
import pandas as pd
import keras.backend as K
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

x_train, y_train, x_valid, y_valid, x_test, y_test = [], [], [], [], [], []
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=0, stratify=y_train)

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_valid /= 255
x_test /= 255

x_train = x_train.reshape(len(x_train), 28, 28)
x_test = x_test.reshape(10000, 28, 28)
x_valid = x_valid.reshape(len(x_valid), 28, 28)

print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'valid samples')
print(x_test.shape[0], 'test samples')

classes_name = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boots"]



def Gaussian_kernal_1D(x):
    sigma_sq = K.constant(0.01445)  # sigma = sqrt(0.5)
    K1 = 1 / (K.sqrt(2 * np.pi * sigma_sq))
    K2 = K.square(x) / (2 * sigma_sq)
    y = K1 * K.exp(-1 * K2)
    return y


def cal_V_J(Y, C, N):
    y = K.expand_dims(Y[:, 0], axis=-1)
    Y0 = y - K.transpose(y)
    Y00 = Gaussian_kernal_1D(Y0)

    for w in range(1,N):
        y1 = K.expand_dims(Y[:, w], axis=-1)
        Y1 = y1 - K.transpose(y1)
        Y11 = Gaussian_kernal_1D(Y1)
        Y00 = Multiply()([Y00, Y11])

    C0 = C-K.transpose(C)
    res = tf.equal(C0, K.constant(0))
    T_0 = K.constant(0)
    T_1 = K.constant(1)
    C1 = tf.where(res, T_1, T_0)
    Y22 = Multiply()([Y00, C1])

    V_J = K.mean(Y22)

    return V_J

def cal_V_C(Y, C, N):
    # Y_w = {}
    y0 = K.expand_dims(Y[:, 0], axis=-1)
    Y0 = y0 - K.transpose(y0)
    Y00 = Gaussian_kernal_1D(Y0)

    for w in range(1,N):
        y1 = K.expand_dims(Y[:, w], axis=-1)
        Y1 = y1 - K.transpose(y1)
        Y11 = Gaussian_kernal_1D(Y1)
        Y00 = Multiply()([Y00, Y11])
        Y2 = Multiply()([Y00, Y11])
    Y22 = K.mean(Y2, axis=1)
    C0 = C-K.transpose(C)
    res = tf.equal(C0, K.constant(0))
    T_0 = K.constant(0)
    T_1 = K.constant(1)
    C1 = tf.where(res, T_1, T_0)
    C00 = K.mean(C1, axis=1)
    Y3 = Multiply()([Y22, C00])

    V_C = K.mean(Y3)
    return V_C


def cal_V_M(Y, C, N):
    y0 = K.expand_dims(Y[:, 0], axis=-1)
    Y0 = y0 - K.transpose(y0)
    Y00 = Gaussian_kernal_1D(Y0)

    for w in range(1, N):
        y1 = K.expand_dims(Y[:, w], axis=-1)
        Y1 = y1 - K.transpose(y1)
        Y11 = Gaussian_kernal_1D(Y1)
        Y00 = Multiply()([Y00, Y11])
        Y2 = Multiply()([Y00, Y11])
    Y22 = K.mean(Y2)

    C0 = C-K.transpose(C)
    res = tf.equal(C0, K.constant(0))
    T_0 = K.constant(0)
    T_1 = K.constant(1)
    C1 = tf.where(res, T_1, T_0)
    C00 = K.mean(C1)
    V_M = Y22*C00
    return V_M

def QMI_fn(y_true, y_pred):
    N = 40
    C_C = y_true
    Y_Y = y_pred
    V_J = cal_V_J(Y_Y, C_C, N)
    V_C = cal_V_C(Y_Y, C_C, N)
    V_M = cal_V_M(Y_Y, C_C, N)
    V = -1*(V_J + V_M - 2 * V_C)

    return V



model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(500, activation='relu', kernel_initializer='he_normal', input_shape=(28, 28, 1)))
model.add(Dropout(0.1))
model.add(Dense(200, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.1))
model.add(Dense(40, activation='sigmoid'))
# optim = keras.optimizers.SGD(lr=0.005, momentum=0.975, decay=2e-06, nesterov=True)
optim = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)#, name='Adam'
model.compile(loss=QMI_fn, optimizer=optim, metrics=['accuracy'])
# model[i].summary()
history = model.fit(x_train, y_train,
                          batch_size=200,
                          epochs=20,
                          verbose=2,
                          validation_data=(x_valid, y_valid))
score_valid = model.evaluate(x_valid, y_valid, verbose=0)
score_train = model.evaluate(x_train, y_train, verbose=0)
# print('Train accuracy:', score_train[1], ', Validation accuracy:',
#         score_valid[1])
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()
y_pred = model.predict_proba(x_test)