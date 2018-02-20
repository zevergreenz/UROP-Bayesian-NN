import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from mnist_bayesian_neural_network import MnistBayesianMultiLayer
from bayesian_cnn import BayesianCNN

def random_sample_active_learning(model, train_x, train_y, unlabelled_x, unlabelled_y, x_test, y_test, iters=50, k=100):
    for i in range(iters):
        print("Active learning iteration %d" % i)
        print("Total data used so far: %d" % train_x.shape[0])
        pred = np.zeros(unlabelled_y.shape)

        idx = np.random.choice(len(unlabelled_x), k, replace=False)

        # Add the selected data points to train set.
        train_x = np.append(train_x, np.take(unlabelled_x, idx, axis=0), axis=0)
        train_y = np.append(train_y, np.take(unlabelled_y, idx, axis=0), axis=0)

        # Train the model again.
        model.optimize(train_x, train_y)

        # Delete the selected data points from the unlabelled set.
        unlabelled_x = np.delete(unlabelled_x, idx, 0)
        unlabelled_y = np.delete(unlabelled_y, idx, 0)

        # Test the accuracy of the model.
        acc = model.validate(x_test, y_test)
        print(np.array(acc).mean())

def maximum_entropy_active_learning(model, train_x, train_y, unlabelled_x, unlabelled_y, x_test, y_test, iters=50, k=100):
    for i in range(iters):
        print("Active learning iteration %d" % i)
        print("Total data used so far: %d" % train_x.shape[0])
        pred = np.zeros(unlabelled_y.shape)

        for _ in range(10):
            pred += model.predict(unlabelled_x)
        pred /= 10

        entropy = np.sum(-1 * pred * np.log(pred + 1e-9), axis=1)
        # entropy = np.sum(-1 * pred * np.log(pred + 1e-9), axis=1)
        idx = np.argpartition(entropy, -k)[-k:]

        # Add the selected data points to train set.
        train_x = np.append(train_x, np.take(unlabelled_x, idx, axis=0), axis=0)
        train_y = np.append(train_y, np.take(unlabelled_y, idx, axis=0), axis=0)

        # Train the model again.
        model.optimize(train_x, train_y)

        # Delete the selected data points from the unlabelled set.
        unlabelled_x = np.delete(unlabelled_x, idx, 0)
        unlabelled_y = np.delete(unlabelled_y, idx, 0)

        # Test the accuracy of the model.
        acc = model.validate(x_test, y_test)
        print(np.array(acc).mean())

# def load_data():
#     mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
#     x_train = mnist.train.images
#     y_train = np.argmax(mnist.train.labels, axis=1)
#     x_test  = mnist.test.images
#     y_test  = np.argmax(mnist.test.labels, axis=1)
#     return (x_train, y_train, x_test, y_test)

def load_data():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    x_train = np.reshape(mnist.train.images, (-1, 28, 28, 1))
    y_train = mnist.train.labels
    x_test  = np.reshape(mnist.test.images, (-1, 28, 28, 1))
    y_test  = mnist.test.labels
    return (x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    # model = MnistBayesianMultiLayer()
    model = BayesianCNN()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x_train, y_train, x_test, y_test = load_data()
        # model.optimize(x_train, y_train)
        train_x = x_train[:100]
        train_y = y_train[:100]
        unlabelled_x = x_train[100:]
        unlabelled_y = y_train[100:]
        # maximum_entropy_active_learning(model, train_x, train_y, unlabelled_x, unlabelled_y, x_test, y_test)
        random_sample_active_learning(model, train_x, train_y, unlabelled_x, unlabelled_y, x_test, y_test)