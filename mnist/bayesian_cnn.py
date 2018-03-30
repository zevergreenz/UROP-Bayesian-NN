import numpy as np
import tensorflow as tf
import keras
import random

from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2
from layers.inference_dropout import InferenceDropout as Dropout
from load_data import load_data

class BayesianCNN(object):

    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        model = Sequential()

        model.add(Conv2D(20, kernel_size=(5, 5), border_mode='same', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, (5, 5), border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # model.compile(loss=keras.losses.kullback_leibler_divergence, 
        #               optimizer=keras.optimizers.Adadelta(), 
        #               metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model = model

    def optimize(self, x, y, epochs=12, batch_size=32):
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, x, batch_size=32):
        return self.model.predict(x, batch_size=batch_size)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=0)

    def sample(self, x, num_samples=30):
        probs = []
        for _ in range(num_samples):
            probs.append(self.model.predict(x))

        predictive_mean = np.mean(probs, axis=0)
        predictive_variance = np.var(probs, axis=0)

        return predictive_mean, predictive_variance


    def validate(self, X_test, Y_test, n_samples=30):
        probs = []
        accuracies = []
        for _ in range(n_samples):
            prob = self.model.predict(X_test)
            pred = np.argmax(prob, axis=1)
            accuracy = (pred == np.argmax(Y_test, axis=1)).mean() * 100
            accuracies.append(accuracy)
        return accuracies

class BayesianCNN2(object):

    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(4, 4), border_mode='valid', activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (4, 4), border_mode='valid', activation='relu'))
        model.add(Conv2D(64, (4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        weight_decay = 2.5 / float(100)
        model.add(Flatten())
        model.add(Dense(128, kernel_regularizer=l2(weight_decay), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # model.compile(loss=keras.losses.kullback_leibler_divergence, 
        #               optimizer=keras.optimizers.Adadelta(), 
        #               metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model = model

    def optimize(self, x, y, epochs=120, batch_size=32):
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, x, batch_size=32):
        return self.model.predict(x, batch_size=batch_size)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=0)

    def sample(self, x, num_samples=30):
        probs = []
        for _ in range(num_samples):
            probs.append(self.model.predict(x))

        predictive_mean = np.mean(probs, axis=0)
        predictive_variance = np.var(probs, axis=0)

        return predictive_mean, predictive_variance


    def validate(self, X_test, Y_test, n_samples=30):
        probs = []
        accuracies = []
        for _ in range(n_samples):
            prob = self.model.predict(X_test)
            pred = np.argmax(prob, axis=1)
            accuracy = (pred == np.argmax(Y_test, axis=1)).mean() * 100
            accuracies.append(accuracy)
        return accuracies

def main(_):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    print("Bayesian CNN accuracy:")
    model = BayesianCNN()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x_train = np.reshape(mnist.train.images, (-1, 28, 28, 1))
        y_train = mnist.train.labels
        x_test = np.reshape(mnist.test.images, (-1, 28, 28, 1))
        y_test = mnist.test.labels
        model.optimize(x_train, y_train)
        acc = model.validate(x_test, y_test)
        print(np.array(acc).mean())

if __name__ == '__main__':
    tf.app.run()