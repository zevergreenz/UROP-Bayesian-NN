import numpy as np
import tensorflow as tf
import pandas as pd
import edward as ed
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal

def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha * x)

class MnistBayesianSingleLayer(object):
    def __init__(self, input_dim=784, output_dim=10, batch_size=100):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        self.w1_shape = [784, 15]
        self.w2_shape = [15, 10]
        self.b1_shape = [15]
        self.b2_shape = [10]
        self.w1 = Normal(loc=tf.zeros(self.w1_shape), scale=tf.ones(self.w1_shape))
        self.w2 = Normal(loc=tf.zeros(self.w2_shape), scale=tf.ones(self.w2_shape))
        self.b1 = Normal(loc=tf.zeros(self.b1_shape), scale=tf.ones(self.b1_shape))
        self.b2 = Normal(loc=tf.zeros(self.b2_shape), scale=tf.ones(self.b2_shape))

        self.X_placeholder = tf.placeholder(tf.float32, (None, self.w1_shape[0]))
        self.Y_placeholder = tf.placeholder(tf.int32, (None,))
        self.categorical = Categorical(self.compute(self.X_placeholder, self.w1, self.b1, self.w2, self.b2))

        self.qw1 = Normal(loc=tf.Variable(tf.random_normal(self.w1_shape)),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.w1_shape))))        
        self.qw2 = Normal(loc=tf.Variable(tf.random_normal(self.w2_shape)),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.w2_shape))))
        self.qb1 = Normal(loc=tf.Variable(tf.random_normal(self.b1_shape)),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.b1_shape))))        
        self.qb2 = Normal(loc=tf.Variable(tf.random_normal(self.b2_shape)),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.b2_shape))))

        self.inference = ed.KLqp({
                self.w1: self.qw1,
                self.w2: self.qw2,
                self.b1: self.qb1,
                self.b2: self.qb2
            }, data={self.categorical: self.Y_placeholder})
        self.inference.initialize(n_iter=100, n_print=100, scale={self.categorical: 50000 / self.batch_size})

    def compute(self, X, w1, b1, w2, b2):
        o1 = leaky_relu(tf.matmul(X, w1) + b1)
        o2 = tf.matmul(o1, w2) + b2
        return o2

    # def optimize(self, mnist):
    #     for _ in range(self.inference.n_iter):
    #         X_batch, Y_batch = mnist.train.next_batch(self.batch_size)
    #         Y_batch = np.argmax(Y_batch, axis=1)
    #         info_dict = self.inference.update(feed_dict={
    #                 self.X_placeholder: X_batch,
    #                 self.Y_placeholder: Y_batch
    #             })
    #         self.inference.print_progress(info_dict)

    def optimize(self, X, Y):
        Y = np.argmax(Y, axis=1)
        for _ in range(self.inference.n_iter):
            info_dict = self.inference.update(feed_dict={
                    self.X_placeholder: X,
                    self.Y_placeholder: Y
                })
            self.inference.print_progress(info_dict)

    def realize_network(self, X):
        w1 = self.qw1.sample()
        w2 = self.qw2.sample()
        b1 = self.qb1.sample()
        b2 = self.qb2.sample()
        return tf.nn.softmax(self.compute(X, w1, b1, w2, b2))

    def predict(self, X):
        return self.realize_network(X).eval()

    def validate(self, mnist, n_samples):
        X_test = mnist.test.images
        Y_test = mnist.test.labels
        Y_test = np.argmax(Y_test, axis=1)
        probs = []
        for _ in range(n_samples):
            prob = self.realize_network(X_test)
            probs.append(prob.eval())
        accuracies = []
        for prob in probs:
            pred = np.argmax(prob, axis=1)
            accuracy = (pred == Y_test).mean() * 100
            accuracies.append(accuracy)
        return accuracies

def MLP(w1, b1, w2, b2, w3, b3, w4, b4, X):
    h = leaky_relu(tf.matmul(X, w1) + b1)
    h = leaky_relu(tf.matmul(h, w2) + b2)
    # h = leaky_relu(tf.matmul(h, w3) + b3)
    # h = tf.matmul(h, w4) + b4
    # h = tf.tanh(tf.matmul(h, w2) + b2)
    # h = tf.tanh(tf.matmul(h, w3) + b3)
    h = tf.matmul(h, w3) + b3
    return h

class MnistBayesianMultiLayer(object):
    def __init__(self, input_dim=784, output_dim=10, batch_size=100):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        self.w1_shape = [784, 800]
        self.w2_shape = [800, 800]
        self.w3_shape = [800, 10]
        self.w4_shape = [10, 10]
        self.b1_shape = [800]
        self.b2_shape = [800]
        self.b3_shape = [10]
        self.b4_shape = [10]

        self.X_placeholder = tf.placeholder(tf.float32, (None, input_dim))
        self.Y_placeholder = tf.placeholder(tf.int32, (None,))

        self.w1 = Normal(loc=tf.zeros(self.w1_shape), scale=tf.ones(self.w1_shape))
        self.w2 = Normal(loc=tf.zeros(self.w2_shape), scale=tf.ones(self.w2_shape))
        self.w3 = Normal(loc=tf.zeros(self.w3_shape), scale=tf.ones(self.w3_shape))
        self.w4 = Normal(loc=tf.zeros(self.w4_shape), scale=tf.ones(self.w4_shape))
        self.b1 = Normal(loc=tf.zeros(self.b1_shape), scale=tf.ones(self.b1_shape))
        self.b2 = Normal(loc=tf.zeros(self.b2_shape), scale=tf.ones(self.b2_shape))
        self.b3 = Normal(loc=tf.zeros(self.b3_shape), scale=tf.ones(self.b3_shape))
        self.b4 = Normal(loc=tf.zeros(self.b4_shape), scale=tf.ones(self.b4_shape))

        # o1 = tf.nn.relu(tf.matmul(self.X_placeholder, self.w1) + self.b1)
        # o2 = tf.nn.relu(tf.matmul(o1, self.w2) + self.b2)
        # o3 = tf.nn.relu(tf.matmul(o2, self.w3) + self.b3)
        # o4 = tf.nn.relu(tf.matmul(o3, self.w4) + self.b4)
        self.categorical = Categorical(MLP(self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4, self.X_placeholder))

        self.qw1 = Normal(loc=tf.Variable(tf.random_normal(self.w1_shape), name='qw1_loc'),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.w1_shape), name='qw1_scale')))
        self.qw2 = Normal(loc=tf.Variable(tf.random_normal(self.w2_shape), name='qw2_loc'),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.w2_shape), name='qw2_scale')))
        self.qw3 = Normal(loc=tf.Variable(tf.random_normal(self.w3_shape), name='qw3_loc'),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.w3_shape), name='qw3_scale')))
        self.qw4 = Normal(loc=tf.Variable(tf.random_normal(self.w4_shape), name='qw4_loc'),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.w4_shape), name='qw4_scale')))
        self.qb1 = Normal(loc=tf.Variable(tf.random_normal(self.b1_shape), name='qb1_loc'),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.b1_shape), name='qb1_scale')))
        self.qb2 = Normal(loc=tf.Variable(tf.random_normal(self.b2_shape), name='qb2_loc'),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.b2_shape), name='qb2_scale')))
        self.qb3 = Normal(loc=tf.Variable(tf.random_normal(self.b3_shape), name='qb3_loc'),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.b3_shape), name='qb3_scale')))
        self.qb4 = Normal(loc=tf.Variable(tf.random_normal(self.b4_shape), name='qb4_loc'),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.b4_shape), name='qb4_scale')))

        self.inference = ed.KLqp({
                self.w1: self.qw1, 
                self.w2: self.qw2,
                self.w3: self.qw3,
                self.w4: self.qw4,
                self.b1: self.qb1,
                self.b2: self.qb2,
                self.b3: self.qb3,
                self.b4: self.qb4
            }, data={self.categorical: self.Y_placeholder})
        self.inference.initialize(n_iter=1, n_print=100, scale={self.categorical: 55000 / self.batch_size})

    # def optimize(self, mnist):
    #     for _ in range(self.inference.n_iter):
    #         X_batch, Y_batch = mnist.train.next_batch(self.batch_size)
    #         Y_batch = np.argmax(Y_batch, axis=1)
    #         info_dict = self.inference.update(feed_dict={
    #                 self.X_placeholder: X_batch,
    #                 self.Y_placeholder: Y_batch
    #             })
    #         self.inference.print_progress(info_dict)

    def optimize(self, X, Y):
        for _ in range(self.inference.n_iter):
            info_dict = self.inference.update(feed_dict={
                    self.X_placeholder: X,
                    self.Y_placeholder: Y
                })
            self.inference.print_progress(info_dict)

    def realize_network(self, X):
        sw1 = self.qw1.sample()
        sw2 = self.qw2.sample()
        sw3 = self.qw3.sample()
        sw4 = self.qw4.sample()
        sb1 = self.qb1.sample()
        sb2 = self.qb2.sample()
        sb3 = self.qb3.sample()
        sb4 = self.qb4.sample()
        return tf.nn.softmax(MLP(sw1, sb1, sw2, sb2, sw3, sb3, sw4, sb4, X))

    def predict(self, X):
        return np.argmax(self.realize_network(X).eval(), axis=1)

    # def validate(self, mnist, n_samples):
    #     X_test = mnist.test.images
    #     Y_test = mnist.test.labels
    #     Y_test = np.argmax(Y_test, axis=1)
    #     probs = []
    #     for _ in range(n_samples):
    #         prob = self.realize_network(X_test)
    #         probs.append(prob.eval())
    #     accuracies = []
    #     for prob in probs:
    #         pred = np.argmax(prob, axis=1)
    #         accuracy = (pred == Y_test).mean() * 100
    #         accuracies.append(accuracy)
    #     return accuracies

    def validate(self, X_test, Y_test, n_samples=30):
        probs = []
        for _ in range(n_samples):
            prob = self.realize_network(X_test)
            probs.append(prob.eval())
        accuracies = []
        for prob in probs:
            pred = np.argmax(prob, axis=1)
            accuracy = (pred == Y_test).mean() * 100
            accuracies.append(accuracy)
        return accuracies

class MnistSingleLayer(object):
    def __init__(self, mnist, input_dim=768, output_dim=10, batch_size=100):
        self.input_dim = 768
        self.output_dim = 10
        self.batch_size = 100

        self.w_shape = [784, 10]
        self.b_shape = [10]
        # self.w = tf.Variable(tf.random_normal(self.w_shape))
        # self.b = tf.Variable(tf.random_normal(self.b_shape))
        self.w = tf.get_variable('w', self.w_shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable('b', self.b_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.))

        self.X_placeholder = tf.placeholder(tf.float32, (None, self.w_shape[0]))
        self.Y_placeholder = tf.placeholder(tf.int32, (None,))
        self.nn = tf.matmul(self.X_placeholder, self.w) + self.b
        self.pred = tf.nn.softmax(self.nn)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.nn, labels=self.Y_placeholder))

        self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        self.train = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss, global_step=self.global_step)

    def optimize(self, mnist, sess):
        for _ in range(250):
            X_train = mnist.train.images
            Y_train = mnist.train.labels
            Y_train = np.argmax(Y_train, axis=1)
            sess.run(self.train, feed_dict={
                    self.X_placeholder: X_train,
                    self.Y_placeholder: Y_train
                })
        # X_batches = []
        # Y_batches = []
        # for i in range(3):
        #     X, Y = mnist.train.next_batch(50)
        #     X_batches.append(X)
        #     Y_batches.append(Y)
        # for i in range(250):
        #     # X_batch, Y_batch = mnist.train.next_batch(self.batch_size)
        #     X_batch = X_batches[i % 3]
        #     Y_batch = Y_batches[i % 3]
        #     Y_batch = np.argmax(Y_batch, axis=1)
        #     sess.run(self.train, feed_dict={
        #             self.X_placeholder: X_batch,
        #             self.Y_placeholder: Y_batch
        #         })

    def validate(self, mnist, sess):
        X_test = mnist.test.images
        Y_test = mnist.test.labels
        Y_test = np.argmax(Y_test, axis=1)
        pred = sess.run(self.pred, feed_dict={
                self.X_placeholder: X_test,
                self.Y_placeholder: Y_test
            })
        pred = np.argmax(pred, axis=1)
        return (pred == Y_test).mean() * 100


# def main(_):
#     mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
#     model = MnistBayesianSingleLayer(mnist)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         model.optimize(mnist)
#         acc = model.validate(mnist, 30)
#         print(acc)

def main(_):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    # print("Single Layer Perceptron accuracy:")
    # model = MnistSingleLayer(mnist)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     model.optimize(mnist, sess)
    #     acc = model.validate(mnist, sess)
    #     print(acc)
    # print("Bayesian Single Layer Perceptron accuracy:")
    # model = MnistBayesianSingleLayer(mnist)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     model.optimize(mnist)
    #     acc = model.validate(mnist, 30)
    #     print(acc)

    print("Bayesian Multi Layer Perceptron accuracy:")
    model = MnistBayesianMultiLayer()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # model.optimize(mnist)
        x_train = mnist.train.images
        y_train = np.argmax(mnist.train.labels, axis=1)
        model.optimize(x_train, y_train)
        acc = model.validate(mnist, 30)
        print(acc)

if __name__ == '__main__':
    tf.app.run()