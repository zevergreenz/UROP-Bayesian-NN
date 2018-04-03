import numpy as np
import tensorflow as tf
import keras
import random

from keras import backend as K
from keras.datasets import mnist
from keras.utils import np_utils, generic_utils

def load_data(cnn=True):
    print("Loading data...")
    (X_train_all, y_train_all), (X_test, y_test) = mnist.load_data()

    if cnn:
        X_train_all = X_train_all.reshape(X_train_all.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    else:
        X_train_all = X_train_all.reshape(X_train_all.shape[0], 784)
        X_test = X_test.reshape(X_test.shape[0], 784)

    # Shuffle the training data.
    random_split = np.asarray(random.sample(range(0, X_train_all.shape[0]), X_train_all.shape[0]))

    X_train_all = X_train_all[random_split]
    y_train_all = y_train_all[random_split]

    X_valid = X_train_all[10000:15000]
    y_valid = y_train_all[10000:15000]

    X_pool = X_train_all[20000:60000]
    y_pool = y_train_all[20000:60000]

    X_train_all = X_train_all[0:10000]
    y_train_all = y_train_all[0:10000]

    X_train = np.array([])
    y_train = np.array([])
    for c in range(10):
        idx = np.array( np.where(y_train_all == c) ).T
        idx = idx[0:10, 0]
        X = X_train_all[idx]
        y = y_train_all[idx]
        if X_train.shape[0] == 0:
            X_train = X
            y_train = y
        else:
            X_train = np.append(X_train, X, axis=0)
            y_train = np.append(y_train, y)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    X_pool = X_pool.astype('float32')
    X_train /= 255
    X_test /= 255
    X_valid /= 255
    X_pool /= 255

    Y_test = np_utils.to_categorical(y_test, 10)
    Y_valid = np_utils.to_categorical(y_valid, 10)
    Y_pool = np_utils.to_categorical(y_pool, 10)
    Y_train = np_utils.to_categorical(y_train, 10)

    return (X_train, Y_train), (X_pool, Y_pool), (X_valid, Y_valid), (X_test, Y_test)
