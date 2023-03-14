from sys import float_info
from math import ceil, floor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn
from sklearn.preprocessing import PolynomialFeatures
np.set_printoptions(suppress=True)
np.random.seed(7)

def mle_estimator(X,y):
    value = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return value
def map_estimator(X, y, gamma):
    value = np.linalg.inv(X.T.dot(X) + (1 / gamma)*np.eye(X.shape[1])).dot(X.T).dot(y)
    return value

def ase(y_pred, y_true):
    error = y_pred - y_true
    mean = np.mean(error ** 2)
    return mean
