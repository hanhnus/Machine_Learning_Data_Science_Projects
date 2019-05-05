# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
import random
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split
from utilities.losses        import compute_loss
from utilities.optimizers    import gradient_descent, pso, mini_batch_gradient_descent
from utilities.visualization import visualize_train, visualize_test


seed = 309
# Freeze the random seed
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.3

# Training settings
alpha = 0.1  # step size
max_iters = 50  # max iterations


def load_data():
    """
    Load Data from CSV
    :return: df    a panda data frame
    """
#    df = pd.read_csv("../data/Part2.csv")
    df = pd.read_csv("../data/Part2Outliers.csv")
#    df = df[df['Height'] > 10]
#    df = df[df['Weight'] < 800]
    return df


def data_preprocess(data):
    """
    Data preprocess:
        1. Split the entire dataset into train and test
        2. Split outputs and inputs
        3. Standardize train and test
        4. Add intercept dummy for computation convenience
    :param data: the given dataset (format: panda DataFrame)
    :return: Xs_train_set       train data contains only inputs
             y_train_set     train data contains only labels
             Xs_test_set        test data contains only inputs
             y_test_set      test data contains only labels
             t_r_ain_s_e_t_fu_ll       train data (full) contains both inputs and labels
             t_e_st_s_e_t_fu_ll       test data (full) contains both inputs and labels
    """
    # Split the data into train and test
    train_set, test_set = train_test_split(data, test_size = train_test_split_test_size)  # (350, 2)  (150, 2)
    # Pre-process data (both train and test)
    Xs_train_set = train_set.drop(["Height"], axis = 1)  # (350, 1)
    y_train_set  = train_set["Height"]                   # (350, 1)
    Xs_test_set  = test_set.drop(["Height"], axis = 1)   # (150, 1)
    y_test_set   = test_set["Height"]                    # (150, 1)

    # Standardize the inputs
    train_mean   = Xs_train_set.mean()
    train_std    = Xs_train_set.std()
    Xs_train_set = (Xs_train_set - train_mean) / train_std
    Xs_test_set  = (Xs_test_set - train_mean)  / train_std

    # Tricks: add dummy intercept to both train and test
    Xs_train_set['intercept_dummy'] = pd.Series(1.0, index = Xs_train_set.index)
    Xs_test_set['intercept_dummy']  = pd.Series(1.0, index = Xs_test_set.index)
    return Xs_train_set, y_train_set, Xs_test_set, y_test_set, train_set, test_set


def learn(y, x, theta, max_iters, alpha, optimizer_type, metric_type):
    """
    Learn to estimate the regression parameters (i.e., w and b)
    :param y:                   train labels
    :param x:                   train data
    :param theta:               model parameter
    :param max_iters:           max training iterations
    :param alpha:               step size
    :param optimizer_type:      optimizer type (default: Batch Gradient Descient): GD, SGD, MiniBGD or PSO
    :param metric_type:         metric type (MSE, RMSE, R2, MAE). NOTE: MAE can't be optimized by GD methods.
    :return: thetas              all updated model parameters tracked during the learning course
             losses             all losses tracked during the learning course
    """
    thetas = None
    losses = None
    if optimizer_type   == "BGD":
        thetas, losses  = gradient_descent(y, x, theta, max_iters, alpha, metric_type)
    elif optimizer_type == "MiniBGD":
        thetas, losses  = mini_batch_gradient_descent(y, x, theta, max_iters, alpha, metric_type, mini_batch_size)
    elif optimizer_type == "PSO":
        thetas, losses  = pso(y, x, theta, max_iters, 100, metric_type)
    else:
        raise ValueError(
            "[ERROR] The optimizer '{ot}' is not defined, please double check and re-run your program.".format(
                ot = optimizer_type))
    return thetas, losses


if __name__ == '__main__':
    # Settings
    metric_type     = "MSE"         # MSE, RMSE, MAE, R2
    optimizer_type  = "PSO"     # PSO, BGD， MiniBGD
    mini_batch_size = 10

    # Step 1: Load Data
    data = load_data()

    # Step 2: Preprocess the data
    Xs_train_set, y_train_set, Xs_test_set, y_test_set, train_set, test_set = data_preprocess(data)

    # Step 3: Learning Start
    theta = np.array([0.0, 0.0])                              # Initialize model parameter

    start_time     = datetime.datetime.now()                  # Track learning starting time
    thetas, losses = learn(y_train_set.values, Xs_train_set.values, theta, max_iters, alpha, optimizer_type, metric_type)
    end_time       = datetime.datetime.now()                  # Track learning ending time
    exection_time  = (end_time - start_time).total_seconds()  # Track execution time

    # Step 4: Results presentation
    print("Learn: execution time={t:.3f} seconds".format(t = exection_time))

    # Build baseline model
    print("R2:  ", -compute_loss(y_test_set.values, Xs_test_set.values, thetas[-1], "R2"))  
    print("MSE: ",  compute_loss(y_test_set.values, Xs_test_set.values, thetas[-1], "MSE"))
    print("RMSE:",  compute_loss(y_test_set.values, Xs_test_set.values, thetas[-1], "RMSE"))
    print("MAE: ",  compute_loss(y_test_set.values, Xs_test_set.values, thetas[-1], "MAE"))

#    visualize_train(train_set, y_train_set, Xs_train_set, thetas, losses, max_iters)
    visualize_test(test_set, Xs_test_set, thetas)
