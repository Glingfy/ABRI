#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Development and optimization of Machine Learning moldes, include RF, XGBoost, and SVR
"""
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt


def mlPerformance(actual, pred):
    """
    Evaluate the performance of machine learning models, including CCC, MAE, and RMSE.
    :param actual: actual values
    :param pred: model predictions
    :return: A dictionary containing CCC, MAE, and RMSE.
    """
    # Check the input data
    if len(actual) != len(pred):
        raise ValueError("The length of 'actual' and 'pred' must be the same.")

    mae = metrics.mean_absolute_error(actual, pred)
    rmse = np.sqrt(metrics.mean_squared_error(actual, pred))

    # Calculate CCC
    mean_actual = np.mean(actual)
    mean_pred = np.mean(pred)
    cov_xy = np.sum((actual - mean_actual) * (pred - mean_pred)) / len(actual)
    var_actual = np.var(actual)
    var_pred = np.var(pred)
    ccc = 2 * cov_xy / (var_actual + var_pred + (mean_actual - mean_pred) ** 2)

    print('Machine learning performance:')
    print(f"CCC: {ccc:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return {"CCC": ccc, "MAE": mae, "RMSE": rmse}


def scatterPlot(actual, pred, name, output_path='output/'):
    """
    Plot a scatter plot of the actual and predicted values and add a 1:1 line
    :param actual: actual values
    :param pred: model predictions
    :param name: File name for saving the graph
    :param output_path: The path to save the chart, default is 'output/'
    """
    # Set the font
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 1.5

    # Create graphics
    fig, axes = plt.subplots(figsize=(7, 7))

    # Draw scatter plots
    axes.scatter(actual, pred, s=90, c=(0.65, 0.85, 0.65, 0.45), edgecolors=(0.47, 0.75, 0.61, 1))

    # Add 1:1 line
    min_val = min(min(actual), min(pred))
    max_val = max(max(actual), max(pred))
    axes.plot([min_val, max_val], [min_val, max_val], color=(0.23, 0.54, 0.54), linestyle='--', linewidth=2)

    # Set the label and title
    axes.set_xlabel('Actual Values')
    axes.set_ylabel('Estimated Values')
    axes.set_title('Scatter Plot of Actual vs Estimated Values', fontsize=20)

    # Set the scale font size
    axes.tick_params(labelsize=18)

    # Display grid
    axes.grid(True, color='#d3d3d3', alpha=0.8)

    # save figure
    plt.savefig(f'{output_path}{name}.jpg', dpi=300, format='jpg', bbox_inches='tight')

    return


def buildRF(X_train, X_valid, y_train, y_valid, n_iter=100):
    """
    Development and optimization of RF model using Bayesian Optimization
    :param X_train: input features in training data
    :param X_valid: input features in validation data
    :param y_train: target variable in training data
    :param y_valid: target variable in validation data
    :param n_iter: Number of optimization iterations
    :return: the best RF model
    """
    def bst_cv_rf(n, max_depth, min_samples_leaf):
        """
        Objective function to be optimized by Bayesian Optimization.
        """
        model = RandomForestRegressor(n_estimators=int(n),
                                      max_depth=int(max_depth),
                                      min_samples_leaf=int(min_samples_leaf),
                                      random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        mae = metrics.mean_absolute_error(y_valid, y_pred)
        return -mae     # Bayesian optimization needs to maximize the objective function, so it returns -MAE

    # Define the parameter range of Bayesian optimization
    pbounds = {
        'n': (50, 2000),
        'max_depth': (3, 10),
        'min_samples_leaf': (1, 10)
    }
    try:
        # Perform Bayesian optimization
        bst_bo = BayesianOptimization(f=bst_cv_rf, pbounds=pbounds, random_state=0)
        bst_bo.maximize(n_iter=n_iter)

        # Get the best hyperparameters
        best_params = bst_bo.max['params']
        best_mae = -bst_bo.max['target']

        # print the best hyperparameters
        print(f"Best Hyperparameters: n_estimators={int(best_params['n'])}, "
              f"max_depth={int(best_params['max_depth'])}, "
              f"min_samples_leaf={int(best_params['min_samples_leaf'])}")
        print(f"Best Validation MAE: {best_mae:.4f}")

        # build the best model
        rf_bst = RandomForestRegressor(n_estimators=int(bst_bo.max['params']['n']),
                                       max_depth=int(bst_bo.max['params']['max_depth']),
                                       min_samples_leaf=int(bst_bo.max['params']['min_samples_leaf']),
                                       random_state=0)

        rf_bst.fit(X_train, y_train)  # Refit the optimal model on the training set
        print('---RF : the construction and optimization of the model are completed---')
        return rf_bst

    except Exception as e:
        print(f"Error during Bayesian Optimization: {e}")
        return None


def buildSVR(X_train, X_valid, y_train, y_valid, n_iter=100):
    """
    Development and optimization of SVR model using Bayesian Optimization
    :param X_train: input features in training data
    :param X_valid: input features in validation data
    :param y_train: target variable in training data
    :param y_valid: target variable in validation data
    :param n_iter: Number of optimization iterations
    :return: the best SVR model
    """
    def bst_cv_svr(gamma, C):
        model = SVR(kernel='rbf', gamma=gamma, C=C)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        mae = metrics.mean_absolute_error(y_valid, y_pred)
        return -mae

    pbounds = {
        'C': (1, 100),
        'gamma': (0.0001, 2)
    }

    try:
        bst_bo = BayesianOptimization(bst_cv_svr, pbounds=pbounds)
        bst_bo.maximize(n_iter=n_iter)

        # Get the best hyperparameters
        best_params = bst_bo.max['params']
        best_mae = -bst_bo.max['target']

        # print the best hyperparameters
        print(f"Best Hyperparameters: gamma={best_params['gamma']:.4f}, C={best_params['C']:.2f}")
        print(f"Best Validation MAE: {best_mae:.4f}")

        # build the best model
        svr_bst = SVR(kernel='rbf', gamma=best_params['gamma'], C=best_params['C'])
        svr_bst.fit(X_train, y_train)
        print('---SVR : the construction and optimization of the model are completed---')
        return svr_bst

    except Exception as e:
        print(f"Error during Bayesian Optimization: {e}")
        return None


def buildXGBoost(X_train, X_valid, y_train, y_valid, n_iter=100):
    """
    Development and optimization of XGBoost model using Bayesian Optimization.
    :param X_train: input features in training data
    :param X_valid: input features in validation data
    :param y_train: target variable in training data
    :param y_valid: target variable in validation data
    :param n_iter: Number of optimization iterations
    :return: the best XGBoost model
    """
    def bst_cv(n, eta, md):
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            subsample=0.8,
            n_estimators=int(n),
            learning_rate=eta,
            max_depth=int(md),
            random_state=0
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        mae = metrics.mean_absolute_error(y_valid, y_pred)
        return -mae

    pbounds = {
        "n": (50, 1000),  # n_estimators
        "eta": (0.01, 0.1),  # learning_rate
        "md": (1, 15)  # max_depth
    }

    try:
        bst_bo = BayesianOptimization(
            f=bst_cv,
            pbounds=pbounds,
            random_state=0,
            verbose=2  # print the optimization process
        )
        bst_bo.maximize(n_iter=n_iter)

        # Get the best hyperparameters
        best_params = bst_bo.max["params"]
        best_mae = -bst_bo.max["target"]

        print(f"Best Hyperparameters: n_estimators={int(best_params['n'])}, "
              f"learning_rate={best_params['eta']:.4f}, max_depth={int(best_params['md'])}")
        print(f"Best Validation MAE: {best_mae:.4f}")

        # build the best model
        xgb_bst = xgb.XGBRegressor(
            objective="reg:squarederror",
            subsample=0.8,
            n_estimators=int(best_params["n"]),
            learning_rate=best_params["eta"],
            max_depth=int(best_params["md"]),
            random_state=0
        )
        xgb_bst.fit(X_train, y_train)

        print("---XGBoost: The construction and optimization of the model are completed---")

        return xgb_bst

    except Exception as e:
        print(f"Error during Bayesian Optimization: {e}")
        return None


def split_data(X, y, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=None):
    """
    Split data into training, validation, and test sets
    :param X: Input features
    :param y: Target variable
    :param train_ratio: Proportion of data for the training set (default: 0.7)
    :param val_ratio: Proportion of data for the validation set (default: 0.1)
    :param test_ratio: Proportion of data for the test set (default: 0.2)
    :param random_state: Random seed for reproducibility (default: None)
    :return: tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
    val_ratio_adjusted = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


# example
filePath = r'path1/documents/file.xlsx'
data = pd.read_excel(filePath, sheet_name='Sheet')
inputs = ['feature1', 'feature2', 'feature3', '...']        # List of names of input features
target = 'name'     # name of target variable
X = data[inputs]
y = data[target]

# Split data into training, validation, and test sets, the data types for X and y are Pandas DataFrame and Series.
X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=42)

outPath = r'path2/documents'
if not os.path.exists(outPath):
    os.makedirs(outPath)
# build RF
rf_best = buildRF(X_train, X_valid, y_train, y_valid, n_iter=1000)
y_pred_rf = rf_best.predict(X_test)
access_rf = mlPerformance(y_test, y_pred_rf)        # Accuracy of the RF model on the test set
scatterPlot(y_test, y_pred_rf, 'RF', outPath)

# build SVR
svr_best = buildSVR(X_train, X_valid, y_train, y_valid, n_iter=1000)
y_pred_svr = svr_best.predict(X_test)
access_svr = mlPerformance(y_test, y_pred_svr)        # Accuracy of the SVR model on the test set
scatterPlot(y_test, y_pred_svr, 'SVR', outPath)

# build XGBoost
xgb_best = buildXGBoost(X_train, X_valid, y_train, y_valid, n_iter=1000)
y_pred_xgb = xgb_best.predict(X_test)
access_xgb = mlPerformance(y_test, y_pred_xgb)        # Accuracy of the XGBoost model on the test set
scatterPlot(y_test, y_pred_xgb, 'XGBoost', outPath)




