import numpy as np
import tensorflow as tf
import pandas as pd
import edward as ed
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal

def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha * x)

def MLP(all_w, all_b, X):
    num_layer = len(all_w)
    h = X
    for i in range(num_layer - 1):
        h = leaky_relu(tf.matmul(h, all_w[i]) + all_b[i])
    h = tf.matmul(h, all_w[-1]) + all_b[-1]
    return h

class MnistBayesianSingleLayer(object):
    def __init__(self, input_dim=784, output_dim=10, batch_size=100):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_layer = 2

        self.w1_shape = [784, 640]
        self.w2_shape = [640, 10]
        self.b1_shape = [640]
        self.b2_shape = [10]
        self.w1 = Normal(loc=tf.zeros(self.w1_shape), scale=tf.ones(self.w1_shape))
        self.w2 = Normal(loc=tf.zeros(self.w2_shape), scale=tf.ones(self.w2_shape))
        self.b1 = Normal(loc=tf.zeros(self.b1_shape), scale=tf.ones(self.b1_shape))
        self.b2 = Normal(loc=tf.zeros(self.b2_shape), scale=tf.ones(self.b2_shape))
        self.all_w = [self.w1, self.w2]
        self.all_b = [self.b1, self.b2]

        self.X_placeholder = tf.placeholder(tf.float32, (None, self.w1_shape[0]))
        self.Y_placeholder = tf.placeholder(tf.int32, (None,))
        self.categorical = Categorical(MLP(self.all_w, self.all_b, self.X_placeholder))

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

        # Inference.
        self.layer_inference = []
        for i in range(self.num_layer):
            inference_dict = {
                self.all_w[i]: self.all_qw[i],
                self.all_b[i]: self.all_qb[i]
            }
            data_dict = {
                self.categorical: self.Y_placeholder
            }
            for j in range(self.num_layer):
                if j != i:
                    data_dict[self.all_w[j]] = self.all_qw[j]
                    data_dict[self.all_b[j]] = self.all_qb[j]
            inference = ed.KLqp(inference_dict, data=data_dict)
            inference.initialize(n_iter=10000)
            self.layer_inference.append(inference)

        inference_dict = {}
        for i in range(self.num_layer):
            inference_dict[self.all_w[i]] = self.all_qw[i]
            inference_dict[self.all_b[i]] = self.all_qb[i]
        data_dict = {self.categorical: self.Y_placeholder}
        self.all_layer_inference = ed.KLqp(inference_dict, data=data_dict)
        self.all_layer_inference.initialize(n_iter=10000)

    def optimize(self, X, Y, layer=None):
        Y = np.argmax(Y, axis=1)
        inference = None
        # if layer == None:
        #     inference = self.all_layer_inference
        # else:
        #     inference = self.layer_inference[layer]
        # for _ in range(1000):
        #     info_dict = inference.update(feed_dict= {self.X_placeholder: X, self.Y_placeholder: Y})
        #     inference.print_progress(info_dict)
        for _ in range(1000):
            for inference in self.layer_inference:
                info_dict = inference.update(feed_dict= {self.X_placeholder: X, self.Y_placeholder: Y})
                inference.print_progress(info_dict)
        print("")

    def realize_network(self, X, layer=None):
        all_w = []
        all_b = []
        if layer == None:
            for i in range(self.num_layer):
                all_w.append(self.all_qw[i].sample())
                all_b.append(self.all_qb[i].sample())
        else:
            for i in range(self.num_layer):
                if i == layer:
                    all_w.append(self.all_qw[i].sample())
                    all_b.append(self.all_qb[i].sample())
                else:
                    all_w.append(self.all_qw[i].loc)
                    all_b.append(self.all_qb[i].loc)
        return tf.nn.softmax(MLP(all_w, all_b, X))

    def print_model_params(self, sess):
        # for i in range(self.num_layer):
        #     print("qW" + str(i) + " : ", end="")
        #     print(self.all_qw[i].loc.eval())
        #     print("")
        #     print("qb" + str(i) + " : ", end="")
        #     print(self.all_qb[i].loc.eval())
        #     print("")
        print(self.qw1.loc.eval())
        print(self.qw2.loc.eval())

    def predict(self, X, layer=None):
        return self.realize_network(X, layer=layer).eval()

    def validate(self, X_test, Y_test, n_samples=30):
        Y_test = np.argmax(Y_test, axis=1)
        accuracies = []
        for _ in range(n_samples):
            prob = self.realize_network(X_test).eval()
            pred = np.argmax(prob, axis=1)
            accuracy = (pred == Y_test).mean() * 100
            accuracies.append(accuracy)
        return accuracies

class MnistBayesianMultiLayer(object):
    def __init__(self, input_dim=784, output_dim=10, batch_size=100):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_layer = 4

        self.w1_shape = [784, 800]
        self.w2_shape = [800, 800]
        self.w3_shape = [800, 10]
        self.w4_shape = [10, 10]
        self.b1_shape = [800]
        self.b2_shape = [800]
        self.b3_shape = [10]
        self.b4_shape = [10]
        self.w1 = Normal(loc=tf.zeros(self.w1_shape), scale=tf.ones(self.w1_shape))
        self.w2 = Normal(loc=tf.zeros(self.w2_shape), scale=tf.ones(self.w2_shape))
        self.w3 = Normal(loc=tf.zeros(self.w3_shape), scale=tf.ones(self.w3_shape))
        self.w4 = Normal(loc=tf.zeros(self.w4_shape), scale=tf.ones(self.w4_shape))
        self.b1 = Normal(loc=tf.zeros(self.b1_shape), scale=tf.ones(self.b1_shape))
        self.b2 = Normal(loc=tf.zeros(self.b2_shape), scale=tf.ones(self.b2_shape))
        self.b3 = Normal(loc=tf.zeros(self.b3_shape), scale=tf.ones(self.b3_shape))
        self.b4 = Normal(loc=tf.zeros(self.b4_shape), scale=tf.ones(self.b4_shape))
        self.all_w = [self.w1, self.w2, self.w3, self.w4]
        self.all_b = [self.b1, self.b2, self.b3, self.b4]

        self.X_placeholder = tf.placeholder(tf.float32, (None, input_dim))
        self.Y_placeholder = tf.placeholder(tf.int32, (None,))
        self.categorical = Categorical(MLP(self.all_w, self.all_b, self.X_placeholder))

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
        self.all_qw = [self.qw1, self.qw2, self.qw3, self.qw4]
        self.all_qb = [self.qb1, self.qb2, self.qb3, self.qb4]

        # Inference.
        self.layer_inference = []
        for i in range(self.num_layer):
            inference_dict = {
                self.all_w[i]: self.all_qw[i],
                self.all_b[i]: self.all_qb[i]
            }
            data_dict = {
                self.categorical: self.Y_placeholder
            }
            for j in range(self.num_layer):
                if j != i:
                    data_dict[self.all_w[j]] = self.all_qw[j]
                    data_dict[self.all_b[j]] = self.all_qb[j]
            inference = ed.KLqp(inference_dict, data=data_dict)
            inference.initialize(n_iter=10000)
            self.layer_inference.append(inference)

        inference_dict = {}
        for i in range(self.num_layer):
            inference_dict[self.all_w[i]] = self.all_qw[i]
            inference_dict[self.all_b[i]] = self.all_qb[i]
        data_dict = {self.categorical: self.Y_placeholder}
        self.all_layer_inference = ed.KLqp(inference_dict, data=data_dict)
        self.all_layer_inference.initialize(n_iter=10000)

    def optimize(self, X, Y, layer=None):
        Y = np.argmax(Y, axis=1)
        inference = None
        if layer == None:
            inference = self.all_layer_inference
        else:
            inference = self.layer_inference[layer]
        for _ in range(1000):
            info_dict = inference.update(feed_dict= {self.X_placeholder: X, self.Y_placeholder: Y})
            inference.print_progress(info_dict)
        print("")

    def realize_network(self, X, layer=None):
        all_w = []
        all_b = []
        if layer == None:
            for i in range(self.num_layer):
                all_w.append(self.all_qw[i].sample())
                all_b.append(self.all_qb[i].sample())
        else:
            for i in range(self.num_layer):
                if i == layer:
                    all_w.append(self.all_qw[i].sample())
                    all_b.append(self.all_qb[i].sample())
                else:
                    all_w.append(self.all_qw[i].loc)
                    all_b.append(self.all_qb[i].loc)
        return tf.nn.softmax(MLP(all_w, all_b, X))

    def predict(self, X, layer=None):
        return self.realize_network(X, layer=layer).eval()

    def validate(self, X_test, Y_test, n_samples=30):
        Y_test = np.argmax(Y_test, axis=1)
        accuracies = []
        for _ in range(n_samples):
            prob = self.realize_network(X_test).eval()
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