import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from bayesian_nn import MnistBayesianMultiLayer, MnistBayesianSingleLayer
from bayesian_cnn import BayesianCNN
from load_data import load_data

def random_sample_active_learning(model, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=10, k=100):
    all_accuracy = np.array([])
    for i in range(iters):
        print("Active learning iteration %d" % i)
        print("Total data used so far: %d" % X_train.shape[0])
        pred = np.zeros(Y_pool.shape)

        idx = np.random.choice(len(X_pool), k, replace=False)

        # Add the selected data points to train set.
        X_train = np.append(X_train, np.take(X_pool, idx, axis=0), axis=0)
        Y_train = np.append(Y_train, np.take(Y_pool, idx, axis=0), axis=0)

        # Train the model again.
        model.optimize(X_train, Y_train)

        # Delete the selected data points from the unlabelled set.
        X_pool = np.delete(X_pool, idx, 0)
        Y_pool = np.delete(Y_pool, idx, 0)

        # Test the accuracy of the model.
        acc = model.validate(X_test, Y_test)
        acc_mean = np.array(acc).mean()
        print(acc_mean)
        all_accuracy = np.append(all_accuracy, acc_mean)
        np.save('./cnn_random.npy', all_accuracy)

def maximum_entropy_active_learning(model, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=10, k=100):
    all_accuracy = np.array([])
    for i in range(iters):
        print("Active learning iteration %d" % i)
        print("Total data used so far: %d" % X_train.shape[0])
        pred = np.zeros(Y_pool.shape)

        for _ in range(50):
            pred += model.predict(X_pool)
        pred /= 50

        entropy = np.sum(-1 * pred * np.log(pred + 1e-9), axis=1)
        # entropy = np.sum(-1 * pred * np.log(pred + 1e-9), axis=1)
        idx = np.argpartition(entropy, -k)[-k:]

        # Add the selected data points to train set.
        X_train = np.append(X_train, np.take(X_pool, idx, axis=0), axis=0)
        Y_train = np.append(Y_train, np.take(Y_pool, idx, axis=0), axis=0)

        # Train the model again.
        model.optimize(X_train, Y_train)

        # Delete the selected data points from the unlabelled set.
        X_pool = np.delete(X_pool, idx, 0)
        Y_pool = np.delete(Y_pool, idx, 0)

        # Test the accuracy of the model.
        acc = model.validate(X_test, Y_test)
        acc_mean = np.array(acc).mean()
        print(acc_mean)
        all_accuracy = np.append(all_accuracy, acc_mean)
        np.save('./cnn_max_entropy.npy', all_accuracy)


def maximum_meanvar_active_learning(model, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=10, k=100):
    all_accuracy = np.array([])
    for i in range(iters):
        print("Active learning iteration %d" % i)
        print("Total data used so far: %d" % X_train.shape[0])
        pred = []

        for _ in range(50):
            pred += [model.predict(X_pool)]

        meanvar = np.sum(np.var(pred, axis=0), axis=1)
        # entropy = np.sum(-1 * pred * np.log(pred + 1e-9), axis=1)
        idx = np.argpartition(meanvar, -k)[-k:]

        # Add the selected data points to train set.
        X_train = np.append(X_train, np.take(X_pool, idx, axis=0), axis=0)
        Y_train = np.append(Y_train, np.take(Y_pool, idx, axis=0), axis=0)

        # Train the model again.
        model.optimize(X_train, Y_train)

        # Delete the selected data points from the unlabelled set.
        X_pool = np.delete(X_pool, idx, 0)
        Y_pool = np.delete(Y_pool, idx, 0)

        # Test the accuracy of the model.
        acc = model.validate(X_test, Y_test)
        acc_mean = np.array(acc).mean()
        print(acc_mean)
        all_accuracy = np.append(all_accuracy, acc_mean)

    np.save('./cnn_max_meanvar.npy', all_accuracy)


def first_layer_maximum_entropy_active_learning(model, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=50, k=100):
    for i in range(iters):
        print("Active learning iteration %d" % i)
        print("Total data used so far: %d" % X_train.shape[0])

        first_layer_out = model.predict_layers(X_pool)[0]
        pred = np.zeros(first_layer_out.shape)
        print(first_layer_out)

        for _ in range(50):
            pred += model.predict_layers(X_pool)[0]
        pred /= 50

        entropy = np.sum(-1 * pred * np.log(pred + 1e-9), axis=1)
        # entropy = np.sum(-1 * pred * np.log(pred + 1e-9), axis=1)
        idx = np.argpartition(entropy, -k)[-k:]

        # Add the selected data points to train set.
        X_train = np.append(X_train, np.take(X_pool, idx, axis=0), axis=0)
        Y_train = np.append(Y_train, np.take(Y_pool, idx, axis=0), axis=0)

        # Train the model again.
        model.optimize(X_train, Y_train)

        # Delete the selected data points from the unlabelled set.
        X_pool = np.delete(X_pool, idx, 0)
        Y_pool = np.delete(Y_pool, idx, 0)

        # Test the accuracy of the model.
        acc = model.validate(X_test, Y_test)
        print(np.array(acc).mean())

if __name__ == "__main__":
    model = BayesianCNN()
    print("Run active learning random.")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data()
        model.optimize(X_train, Y_train)
        random_sample_active_learning(model, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

    # model = MnistBayesianSingleLayer()
    # print("Run active learning maximum entropy.")
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     x_train, y_train, X_test, Y_test = load_data()
    #     X_pool = x_train[100:]
    #     Y_pool = y_train[100:]
    #     X_train = x_train[:200]
    #     Y_train = y_train[:200]
    #     model.optimize(X_train, Y_train)
    #     maximum_entropy_active_learning(model, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)
    #     # first_layer_maximum_entropy_active_learning(model, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

    # model = MnistBayesianSingleLayer()
    # print("Run active learning maximum mean variance.")
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     x_train, y_train, X_test, Y_test = load_data()
    #     X_pool = x_train[100:]
    #     Y_pool = y_train[100:]
    #     X_train = x_train[:200]
    #     Y_train = y_train[:200]
    #     model.optimize(X_train, Y_train)
    #     maximum_meanvar_active_learning(model, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)
    #     # first_layer_maximum_entropy_active_learning(model, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)