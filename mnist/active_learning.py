import gc
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from bayesian_nn import MnistBayesianMultiLayer, MnistBayesianSingleLayer
from bayesian_cnn import BayesianCNN, BayesianCNN2
from load_data import load_data

def replace_zeroes_for_log(vector):
    new_vector = np.copy(vector)
    small_value = 1
    new_vector[new_vector == 0] = small_value
    return new_vector

def random_sample_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=10, k=50):
    all_data_size = np.array([])
    all_accuracy = np.array([])
    for i in range(iters):
        print("Active learning iteration %d" % i)
        print("Total data used so far: %d" % X_train.shape[0])
        # model.print_model_params(sess)
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

        entropy = np.sum(-1 * pred * np.log2(pred + 1e-9), axis=1)
        # entropy = np.sum(-1 * pred * np.log2(pred + 1e-9), axis=1)
        idx = np.argpartition(entropy, -k)[-k:]

        # Add the selected data points to train set.
        X_train = np.append(X_train, np.take(X_pool, idx, axis=0), axis=0)
        Y_train = np.append(Y_train, np.take(Y_pool, idx, axis=0), axis=0)

        # Initalize new model and train the model again.
        model = BayesianCNN2()
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
        np.save('./cnn_max_entropy.npy', all_accuracy)


def maximum_meanvar_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=10, k=50):
    all_accuracy = np.array([])
    for i in range(iters):
        print("Active learning iteration %d" % i)
        print("Total data used so far: %d" % X_train.shape[0])
        pred = []

        for _ in range(50):
            prediction = model.predict(X_pool)
            add_noise_to_zeros(prediction)
            pred.append(prediction)

        meanvar = np.sum(np.var(pred, axis=0), axis=1)
        # entropy = np.sum(-1 * pred * np.log2(pred + 1e-9), axis=1)
        idx = np.argpartition(meanvar, -k)[-k:]

        # Add the selected data points to train set.
        X_train = np.append(X_train, np.take(X_pool, idx, axis=0), axis=0)
        Y_train = np.append(Y_train, np.take(Y_pool, idx, axis=0), axis=0)

        # Initalize new model and train the model again.
        model = BayesianCNN2()
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

    np.save('./cnn_max_meanvar.npy', all_accuracy)

def BALD_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=10, k=50):
    all_accuracy = np.array([])
    for i in range(iters):
        print("Active learning iteration %d" % i)
        print("Total data used so far: %d" % X_train.shape[0])
        pred = []

        for _ in range(50):
            pred += [model.predict(X_pool)]

        avg_of_entropy = np.sum([np.sum(- p_i * np.log2(p_i), axis=1) for p_i in pred], axis=0) / 50
        pred_avg = np.sum(pred, axis=0) / 50
        entropy_of_avg = np.sum(- pred_avg * np.log2(pred_avg), axis=1)

        bald = entropy_of_avg - avg_of_entropy

        idx = np.argpartition(bald, -k)[-k:]

        # Add the selected data points to train set.
        X_train = np.append(X_train, np.take(X_pool, idx, axis=0), axis=0)
        Y_train = np.append(Y_train, np.take(Y_pool, idx, axis=0), axis=0)

        # Initalize new model and train the model again.
        model = BayesianCNN2()
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

    np.save('./cnn_max_bald.npy', all_accuracy)

def BALD_layer_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=10, k=50):
    all_accuracy = np.array([])
    for i in range(iters):
        print("Active learning iteration %d" % i)
        print("Total data used so far: %d" % X_train.shape[0])

        bald_layers = []
        bald_idxes  = []
        for layer in range(model.num_layer):
            print("Processing layer %d" % layer)
            pred = []
            for _ in range(20):
                pred += [model.predict(X_pool, layer=layer)]
            avg_of_entropy = np.sum([np.sum(- p_i * np.log2(replace_zeroes_for_log(p_i)), axis=1) for p_i in pred], axis=0) / 20
            pred_avg = np.sum(pred, axis=0) / 20
            entropy_of_avg = np.sum(- pred_avg * np.log2(replace_zeroes_for_log(pred_avg)), axis=1)

            bald = entropy_of_avg - avg_of_entropy

            idx = np.argpartition(bald, -k)[-k:]
            bald_layers.append(np.sum(bald[idx]))
            bald_idxes.append(idx)
            gc.collect()

        chosen_layer = np.argmax(bald_layers)
        idx = bald_idxes[chosen_layer]
        print("HERE1")
        print(bald_layers)
        print(bald_idxes)
        print("Choose layer %d to do inference." % chosen_layer)
        print("HERE2")

        # Add the selected data points to train set.
        X_train = np.append(X_train, np.take(X_pool, idx, axis=0), axis=0)
        Y_train = np.append(Y_train, np.take(Y_pool, idx, axis=0), axis=0)

        # Initalize new model and train the model again.
        model = MnistBayesianSingleLayer()
        sess.run(tf.global_variables_initializer())
        model.optimize(X_train, Y_train, layer=None)

        # Delete the selected data points from the unlabelled set.
        X_pool = np.delete(X_pool, idx, 0)
        Y_pool = np.delete(Y_pool, idx, 0)

        # Test the accuracy of the model.
        acc = model.validate(X_test, Y_test)
        acc_mean = np.array(acc).mean()
        print(acc_mean)
        all_accuracy = np.append(all_accuracy, acc_mean)
        gc.collect()

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

#         entropy = np.sum(-1 * pred * np.log2(pred + 1e-9), axis=1)
#         # entropy = np.sum(-1 * pred * np.log2(pred + 1e-9), axis=1)
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
    with tf.Session() as sess:
        print("Run active learning random.")
        model = MnistBayesianSingleLayer()
        sess.run(tf.global_variables_initializer())
        (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data(cnn=False)
        model.optimize(X_train, Y_train)
        random_sample_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=50, k=50)

    # tf.reset_default_graph()
    # with tf.Session() as sess:
    #     print("Run active learning maximum entropy.")
    #     model = MnistBayesianSingleLayer()
    #     sess.run(tf.global_variables_initializer())
    #     (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data(cnn=False)
    #     model.optimize(X_train, Y_train)
    #     maximum_entropy_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

    # tf.reset_default_graph()
    # with tf.Session() as sess:
    #     print("Run active learning maximum meanvar.")
    #     model = MnistBayesianSingleLayer()
    #     sess.run(tf.global_variables_initializer())
    #     (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data(cnn=False)
    #     model.optimize(X_train, Y_train)
    #     maximum_meanvar_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

    # tf.reset_default_graph()
    # with tf.Session() as sess:
    #     print("Run active learning maximum bald.")
    #     model = MnistBayesianSingleLayer()
    #     sess.run(tf.global_variables_initializer())
    #     (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data(cnn=False)
    #     model.optimize(X_train, Y_train)
    #     BALD_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

    # tf.reset_default_graph()
    # with tf.Session() as sess:
    #     print("Run active learning maximum layer bald.")
    #     model = MnistBayesianSingleLayer()
    #     # model = MnistBayesianMultiLayer()
    #     sess.run(tf.global_variables_initializer())
    #     (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data(cnn=False)
    #     model.optimize(X_train, Y_train)
    #     print("Base Model Accuracy: ", np.array(model.validate(X_test, Y_test)).mean())
    #     BALD_layer_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

    # tf.reset_default_graph()
    # with tf.Session() as sess:
    #     print("Run active learning random.")
    #     model = BayesianCNN2()
    #     sess.run(tf.global_variables_initializer())
    #     (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data(cnn=True)
    #     model.optimize(X_train, Y_train)
    #     random_sample_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

    # tf.reset_default_graph()
    # with tf.Session() as sess:
    #     print("Run active learning maximum entropy.")
    #     model = BayesianCNN2()
    #     sess.run(tf.global_variables_initializer())
    #     (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data(cnn=True)
    #     model.optimize(X_train, Y_train)
    #     maximum_entropy_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

    # tf.reset_default_graph()
    # with tf.Session() as sess:
    #     print("Run active learning maximum meanvar.")
    #     model = BayesianCNN2()
    #     sess.run(tf.global_variables_initializer())
    #     (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data(cnn=True)
    #     model.optimize(X_train, Y_train)
    #     maximum_meanvar_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

    # tf.reset_default_graph()
    # with tf.Session() as sess:
    #     print("Run active learning maximum bald.")
    #     model = BayesianCNN2()
    #     sess.run(tf.global_variables_initializer())
    #     (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data(cnn=True)
    #     model.optimize(X_train, Y_train)
    #     BALD_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)

    # tf.reset_default_graph()
    # with tf.Session() as sess:
    #     print("Run active learning maximum layer bald.")
    #     model = BayesianCNN2()
    #     sess.run(tf.global_variables_initializer())
    #     (X_train, Y_train), (X_pool, Y_pool), (_, _), (X_test, Y_test) = load_data(cnn=True)
    #     model.optimize(X_train, Y_train)
    #     BALD_layer_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test)