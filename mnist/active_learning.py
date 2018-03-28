import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from bayesian_nn import MnistBayesianMultiLayer, MnistBayesianSingleLayer
from bayesian_cnn import BayesianCNN
from load_data import load_data

def random_sample_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=10, k=50):
    all_data_size = np.array([])
    all_accuracy = np.array([])
    for i in range(iters):
        print("Active learning iteration %d" % i)
        print("Total data used so far: %d" % X_train.shape[0])
        pred = np.zeros(Y_pool.shape)

        idx = np.random.choice(len(X_pool), k, replace=False)

        # Add the selected data points to train set.
        X_train = np.append(X_train, np.take(X_pool, idx, axis=0), axis=0)
        Y_train = np.append(Y_train, np.take(Y_pool, idx, axis=0), axis=0)

        # Initalize new model and train the model again.
        model = MnistBayesianSingleLayer()
        sess.run(tf.global_variables_initializer())
        model.optimize(X_train, Y_train)

        # Delete the selected data points from the unlabelled set.
        X_pool = np.delete(X_pool, idx, 0)
        Y_pool = np.delete(Y_pool, idx, 0)

        # Test the accuracy of the model.
        acc = model.validate(X_test, Y_test)
        acc_mean = np.array(acc).mean()
        print(acc_mean)
        all_accuracy = np.append(all_accuracy, acc_mean)
        all_data_size = np.append(all_data_size, X_train.shape[0])
        np.save('./nn_random.npy', [all_data_size, all_accuracy])
    print("Total data used: ", all_data_size)
    print("Accuracies: ", all_accuracy)

def maximum_entropy_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=10, k=50):
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

        # Initalize new model and train the model again.
        model = MnistBayesianSingleLayer()
        sess.run(tf.global_variables_initializer())
        model.optimize(X_train, Y_train)

        # Delete the selected data points from the unlabelled set.
        X_pool = np.delete(X_pool, idx, 0)
        Y_pool = np.delete(Y_pool, idx, 0)

        # Test the accuracy of the model.
        acc = model.validate(X_test, Y_test)
        acc_mean = np.array(acc).mean()
        print(acc_mean)
        all_accuracy = np.append(all_accuracy, acc_mean)
        np.save('./nn_max_entropy.npy', all_accuracy)


def maximum_meanvar_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=10, k=50):
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

        # Initalize new model and train the model again.
        model = MnistBayesianSingleLayer()
        sess.run(tf.global_variables_initializer())
        model.optimize(X_train, Y_train)

        # Delete the selected data points from the unlabelled set.
        X_pool = np.delete(X_pool, idx, 0)
        Y_pool = np.delete(Y_pool, idx, 0)

        # Test the accuracy of the model.
        acc = model.validate(X_test, Y_test)
        acc_mean = np.array(acc).mean()
        print(acc_mean)
        all_accuracy = np.append(all_accuracy, acc_mean)

    np.save('./nn_max_meanvar.npy', all_accuracy)

def BALD_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=10, k=50):
    all_accuracy = np.array([])
    for i in range(iters):
        print("Active learning iteration %d" % i)
        print("Total data used so far: %d" % X_train.shape[0])
        pred = []

        for _ in range(50):
            pred += [model.predict(X_pool)]

        avg_of_entropy = np.sum([np.sum(- p_i * np.log(p_i), axis=1) for p_i in pred], axis=0) / 50
        pred_avg = np.sum(pred, axis=0) / 50
        entropy_of_avg = np.sum(- pred_avg * np.log(pred_avg), axis=1)

        bald = entropy_of_avg - avg_of_entropy

        idx = np.argpartition(bald, -k)[-k:]

        # Add the selected data points to train set.
        X_train = np.append(X_train, np.take(X_pool, idx, axis=0), axis=0)
        Y_train = np.append(Y_train, np.take(Y_pool, idx, axis=0), axis=0)

        # Initalize new model and train the model again.
        model = MnistBayesianSingleLayer()
        sess.run(tf.global_variables_initializer())
        model.optimize(X_train, Y_train)

        # Delete the selected data points from the unlabelled set.
        X_pool = np.delete(X_pool, idx, 0)
        Y_pool = np.delete(Y_pool, idx, 0)

        # Test the accuracy of the model.
        acc = model.validate(X_test, Y_test)
        acc_mean = np.array(acc).mean()
        print(acc_mean)
        all_accuracy = np.append(all_accuracy, acc_mean)

    np.save('./nn_max_bald.npy', all_accuracy)

def BALD_layer_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=10, k=50):
    all_accuracy = np.array([])
    for i in range(iters):
        print("Active learning iteration %d" % i)
        print("Total data used so far: %d" % X_train.shape[0])
        pred = []

        for _ in range(50):
            pred += [model.predict(X_pool, layer=1)]

        avg_of_entropy = np.sum([np.sum(- p_i * np.log(p_i), axis=1) for p_i in pred], axis=0) / 50
        pred_avg = np.sum(pred, axis=0) / 50
        entropy_of_avg = np.sum(- pred_avg * np.log(pred_avg), axis=1)

        bald = entropy_of_avg - avg_of_entropy

        idx = np.argpartition(bald, -k)[-k:]

        # Add the selected data points to train set.
        X_train = np.append(X_train, np.take(X_pool, idx, axis=0), axis=0)
        Y_train = np.append(Y_train, np.take(Y_pool, idx, axis=0), axis=0)

        # Initalize new model and train the model again.
        model = MnistBayesianSingleLayer()
        sess.run(tf.global_variables_initializer())
        model.optimize(X_train, Y_train)

        # Delete the selected data points from the unlabelled set.
        X_pool = np.delete(X_pool, idx, 0)
        Y_pool = np.delete(Y_pool, idx, 0)

        # Test the accuracy of the model.
        acc = model.validate(X_test, Y_test)
        acc_mean = np.array(acc).mean()
        print(acc_mean)
        all_accuracy = np.append(all_accuracy, acc_mean)

    np.save('./nn_max_layer_bald.npy', all_accuracy)


# def first_layer_maximum_entropy_active_learning(model, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=50, k=50):
#     for i in range(iters):
#         print("Active learning iteration %d" % i)
#         print("Total data used so far: %d" % X_train.shape[0])

#         first_layer_out = model.predict_layers(X_pool)[0]
#         pred = np.zeros(first_layer_out.shape)
#         print(first_layer_out)

#         for _ in range(50):
#             pred += model.predict_layers(X_pool)[0]
#         pred /= 50

#         entropy = np.sum(-1 * pred * np.log(pred + 1e-9), axis=1)
#         # entropy = np.sum(-1 * pred * np.log(pred + 1e-9), axis=1)
#         idx = np.argpartition(entropy, -k)[-k:]

#         # Add the selected data points to train set.
#         X_train = np.append(X_train, np.take(X_pool, idx, axis=0), axis=0)
#         Y_train = np.append(Y_train, np.take(Y_pool, idx, axis=0), axis=0)

#         # Train the model again.
#         model.optimize(X_train, Y_train)

#         # Delete the selected data points from the unlabelled set.
#         X_pool = np.delete(X_pool, idx, 0)
#         Y_pool = np.delete(Y_pool, idx, 0)

#         # Test the accuracy of the model.
#         acc = model.validate(X_test, Y_test)
#         print(np.array(acc).mean())

if __name__ == "__main__":
    # with tf.Session() as sess:
    #     print("Run active learning random.")
    #     model = MnistBayesianSingleLayer()
    #     sess.run(tf.global_variables_initializer())
    #     (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data(cnn=False)
    #     model.optimize(X_train, Y_train)
    #     random_sample_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

    # tf.reset_default_graph()
    # with tf.Session() as sess:
    #     print("Run active learning maximum entropy.")
    #     model = MnistBayesianSingleLayer()
    #     sess.run(tf.global_variables_initializer())
    #     (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data(cnn=False)
    #     model.optimize(X_train, Y_train)
    #     maximum_entropy_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

    tf.reset_default_graph()
    with tf.Session() as sess:
        print("Run active learning maximum meanvar.")
        model = MnistBayesianSingleLayer()
        sess.run(tf.global_variables_initializer())
        (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data(cnn=False)
        model.optimize(X_train, Y_train)
        maximum_meanvar_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

    tf.reset_default_graph()
    with tf.Session() as sess:
        print("Run active learning maximum bald.")
        model = MnistBayesianSingleLayer()
        sess.run(tf.global_variables_initializer())
        (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data(cnn=False)
        model.optimize(X_train, Y_train)
        BALD_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

    # tf.reset_default_graph()
    # with tf.Session() as sess:
    #     print("Run active learning maximum layer bald.")
    #     model = MnistBayesianSingleLayer()
    #     sess.run(tf.global_variables_initializer())
    #     (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data(cnn=False)
    #     model.optimize(X_train, Y_train)
    #     BALD_layer_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

        # print("Run active learning random.")
        # model = BayesianCNN()
        # (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data()
        # model.optimize(X_train, Y_train)
        # random_sample_active_learning(model, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

        # print("Run active learning maximum entropy.")
        # model = BayesianCNN()
        # (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data()
        # model.optimize(X_train, Y_train)
        # maximum_entropy_active_learning(model, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

        # print("Run active learning maximum meanvar.")
        # model = BayesianCNN()
        # (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data()
        # model.optimize(X_train, Y_train)
        # maximum_meanvar_active_learning(model, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

        # print("Run active learning maximum bald.")
        # model = BayesianCNN()
        # (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data()
        # model.optimize(X_train, Y_train)
        # BALD_active_learning(model, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)