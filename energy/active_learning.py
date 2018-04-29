import gc
import numpy as np
import tensorflow as tf

from bayesian_nn import EEBayesianSingleLayer, EESingleLayer
from load_data import load_data

def replace_zeroes_for_log(vector):
    new_vector = np.copy(vector)
    small_value = 1
    new_vector[new_vector == 0] = small_value
    return new_vector

def random_sample_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=10, k=5):
    all_data_size = np.array([])
    all_accuracy = np.array([])
    for i in range(iters):
        print("Active learning iteration %d" % i)
        print("Total data used so far: %d" % X_train.shape[0])
        # model.print_model_params(sess)

        idx = np.random.choice(len(X_pool), k, replace=False)

        # Add the selected data points to train set.
        X_train = np.append(X_train, np.take(X_pool, idx, axis=0), axis=0)
        Y_train = np.append(Y_train, np.take(Y_pool, idx, axis=0), axis=0)
        gc.collect()

        # Initalize new model and train the model again.
        model = EEBayesianSingleLayer()
        sess.run(tf.global_variables_initializer())
        model.optimize(X_train, Y_train)

        # Delete the selected data points from the unlabelled set.
        X_pool = np.delete(X_pool, idx, 0)
        Y_pool = np.delete(Y_pool, idx, 0)
        gc.collect()

        # Test the accuracy of the model.
        mses = model.validate(X_test, Y_test)
        mses_mean = np.array(mses).mean()
        print(mses_mean)
        all_accuracy = np.append(all_accuracy, mses_mean)
        all_data_size = np.append(all_data_size, X_train.shape[0])
        np.save('./nn_random.npy', [all_data_size, all_accuracy])
        gc.collect()
    print("Total data used: ", all_data_size)
    print("Accuracies: ", all_accuracy)

def maximum_meanvar_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=10, k=5):
    all_accuracy = np.array([])
    for i in range(iters):
        print("Active learning iteration %d" % i)
        print("Total data used so far: %d" % X_train.shape[0])
        pred = []

        for _ in range(100):
            prediction = model.predict(X_pool)
            pred.append(prediction)

        meanvar = np.var(pred, axis=0)
        meanvar = np.reshape(meanvar, (meanvar.shape[0],))
        idx = np.argpartition(meanvar, -k)[-k:]

        # Add the selected data points to train set.
        X_train = np.append(X_train, np.take(X_pool, idx, axis=0), axis=0)
        Y_train = np.append(Y_train, np.take(Y_pool, idx, axis=0), axis=0)

        # Initalize new model and train the model again.
        model = EEBayesianSingleLayer()
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

    np.save('./nn_max_meanvar_500_samples_2.npy', all_accuracy)


if __name__ == "__main__":
    with tf.Session() as sess:
        print("Run active learning random.")
        model = EEBayesianSingleLayer()
        sess.run(tf.global_variables_initializer())
        (X_train, Y_train), (X_pool, Y_pool), (X_test, Y_test) = load_data(cnn=False)
        model.optimize(X_train, Y_train)
        random_sample_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=50, k=3)

    # tf.reset_default_graph()
    # with tf.Session() as sess:
    #     print("Run active learning maximum meanvar.")
    #     model = EEBayesianSingleLayer()
    #     sess.run(tf.global_variables_initializer())
    #     (X_train, Y_train), (X_pool, Y_pool), (X_test, Y_test) = load_data(cnn=False)
    #     model.optimize(X_train, Y_train)
    #     maximum_meanvar_active_learning(model, sess, X_train, Y_train, X_pool, Y_pool, X_test, Y_test, iters=10, k=2)

    # mse = []
    # for _ in range(20):
    #     tf.reset_default_graph()
    #     with tf.Session() as sess:
    #         model = EEBayesianSingleLayer()
    #         # model = EESingleLayer()
    #         sess.run(tf.global_variables_initializer())
    #         (X_train, Y_train), (X_pool, Y_pool), (X_test, Y_test) = load_data(cnn=False)
    #         X_train = np.append(X_train, X_pool, axis=0)
    #         Y_train = np.append(Y_train, Y_pool, axis=0)
    #         model.optimize(X_train, Y_train)
    #         e = model.validate(X_test, Y_test)
    #         mse.append(e)
    #         print(e)
    # print(np.mean(mse))