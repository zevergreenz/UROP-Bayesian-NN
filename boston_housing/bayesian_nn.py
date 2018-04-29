import numpy as np
import tensorflow as tf
import pandas as pd
import edward as ed
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal

def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha * x)

def MLP(all_w, all_b, X):
    num_layer = len(all_w)
    h = X
    for i in range(num_layer - 1):
        # h = tf.nn.relu(tf.matmul(h, all_w[i]) + all_b[i])
        h = leaky_relu(tf.matmul(h, all_w[i]) + all_b[i])
    h = tf.matmul(h, all_w[-1]) + all_b[-1]
    return h

class BHBayesianSingleLayer(object):
    def __init__(self, input_dim=13, output_dim=1, batch_size=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_layer = 2

        self.w1_shape = [input_dim, 50]
        self.w2_shape = [50, output_dim]
        self.b1_shape = [50]
        self.b2_shape = [output_dim]
        self.w1 = Normal(loc=tf.zeros(self.w1_shape), scale=tf.ones(self.w1_shape))
        self.w2 = Normal(loc=tf.zeros(self.w2_shape), scale=tf.ones(self.w2_shape))
        self.b1 = Normal(loc=tf.zeros(self.b1_shape), scale=tf.ones(self.b1_shape))
        self.b2 = Normal(loc=tf.zeros(self.b2_shape), scale=tf.ones(self.b2_shape))
        self.all_w = [self.w1, self.w2]
        self.all_b = [self.b1, self.b2]

        self.X_placeholder = tf.placeholder(tf.float32, (None, self.w1_shape[0]))
        self.Y_placeholder = tf.placeholder(tf.float32, (None))
        self.categorical = Normal(loc=MLP(self.all_w, self.all_b, self.X_placeholder), scale=tf.ones(tf.shape(self.Y_placeholder)))

        self.qw1 = Normal(loc=tf.Variable(tf.random_normal(self.w1_shape)),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.w1_shape))))        
        self.qw2 = Normal(loc=tf.Variable(tf.random_normal(self.w2_shape)),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.w2_shape))))
        self.qb1 = Normal(loc=tf.Variable(tf.random_normal(self.b1_shape)),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.b1_shape))))        
        self.qb2 = Normal(loc=tf.Variable(tf.random_normal(self.b2_shape)),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.b2_shape))))
        self.all_qw = [self.qw1, self.qw2]
        self.all_qb = [self.qb1, self.qb2]

        self.all_layer_inference = ed.KLqp({
            self.w1: self.qw1,
            self.w2: self.qw2,
            self.b1: self.qb1,
            self.b2: self.qb2
        }, data={
            self.categorical: self.Y_placeholder
        })
        self.all_layer_inference.initialize(n_samples=16, n_iter=20000)

        self.y_post = ed.copy(self.categorical, {
            self.w1: self.qw1,
            self.w2: self.qw2,
            self.b1: self.qb1,
            self.b2: self.qb2
        })

    def optimize(self, X, Y, layer=None):
        # Normalize the data.
        print(X.shape)
        self.std_X_train = np.std(X, 0)
        self.std_X_train[self.std_X_train == 0] = 1
        self.mean_X_train = np.mean(X, 0)

        X = (X - np.full(X.shape, self.mean_X_train)) / np.full(X.shape, self.std_X_train)

        self.std_y_train = np.std(Y)
        self.mean_y_train = np.mean(Y)

        Y = (Y - self.mean_y_train) / self.std_y_train

        inference = self.all_layer_inference
        for _ in range(inference.n_iter):
            info_dict = inference.update(feed_dict= {self.X_placeholder: X, self.Y_placeholder: Y})
            inference.print_progress(info_dict)
        print("")

    def realize_network(self, X, layer=None):
        all_w = []
        all_b = []
        for i in range(self.num_layer):
            all_w.append(self.all_qw[i].sample())
            all_b.append(self.all_qb[i].sample())
        return MLP(all_w, all_b, X)

    def predict(self, X, layer=None):
        return self.realize_network(X, layer=layer).eval()

    def validate(self, X_test, Y_test, n_samples=50):
        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / np.full(X_test.shape, self.std_X_train)
        mses = []
        for _ in range(n_samples):
            pred = self.predict(X_test) * self.std_y_train + self.mean_y_train
            pred = pred[:, 0]
            mse = np.mean((Y_test - pred) ** 2)
            mses.append(mse)
        return np.mean(mses)
        # mse = ed.evaluate('mean_squared_error', data={self.X_placeholder: X_test, self.y_post: Y_test, self.Y_placeholder: Y_test})
        # print("Mean square error on test data: ", mse)
        # return mse

class BHSingleLayer(object):
    def __init__(self, input_dim=13, output_dim=1, batch_size=1):
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=13, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(1, kernel_initializer='normal'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def optimize(self, X, Y):
        # Normalize the data.
        self.std_X_train = np.std(X, 0)
        self.std_X_train[self.std_X_train == 0] = 1
        self.mean_X_train = np.mean(X, 0)

        X = (X - np.full(X.shape, self.mean_X_train)) / np.full(X.shape, self.std_X_train)

        self.std_y_train = np.std(Y)
        self.mean_y_train = np.mean(Y)

        Y = (Y - self.mean_y_train) / self.std_y_train
        self.model.fit(X, Y, batch_size=8, epochs=300)

    def validate(self, X_test, Y_test):
        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / np.full(X_test.shape, self.std_X_train)
        pred = self.model.predict(X_test) * self.std_y_train + self.mean_y_train
        mse = np.mean((pred - Y_test) ** 2)
        return mse